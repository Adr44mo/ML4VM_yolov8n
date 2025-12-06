"""
Modified loss computation for YOLOv8 with pose detection
Adds pose classification loss to the standard bbox + cls + dfl losses
"""

import torch
import torch.nn as nn
from utils.util import make_anchors, BoxLoss, Assigner


class ComputeLossWithPose:
    """
    Compute loss for YOLOv8 with pose detection
    
    Returns 4 losses: box_loss, cls_loss, dfl_loss, pose_loss
    """
    
    def __init__(self, model, params, num_poses=3):
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device

        m = model.head  # HeadWithPose module

        self.params = params
        self.stride = m.stride
        self.nc = m.nc  # number of object classes
        self.np = num_poses  # number of pose classes
        self.no = m.no  # original number of outputs (bbox + cls)
        self.reg_max = m.ch
        self.device = device

        self.box_loss = BoxLoss(m.ch - 1).to(device)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.pose_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.assigner = Assigner(nc=self.nc, top_k=10, alpha=0.5, beta=6.0)

        self.project = torch.arange(m.ch, dtype=torch.float, device=device)

    def box_decode(self, anchor_points, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4)
        pred_dist = pred_dist.softmax(3)
        pred_dist = pred_dist.matmul(self.project.type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, -1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        return torch.cat(tensors=(x1y1, x2y2), dim=-1)

    def __call__(self, outputs, targets):
        """
        Args:
            outputs: tuple of (box_outputs, cls_outputs, pose_outputs) from HeadWithPose
            targets: dict with keys:
                - 'cls': object class labels
                - 'box': bounding boxes
                - 'idx': batch indices
                - 'pose': pose labels (0=lying, 1=sitting, 2=standing)
        
        Returns:
            loss_box, loss_cls, loss_dfl, loss_pose
        """
        box_outputs, cls_outputs, pose_outputs = outputs
        
        # Concatenate predictions from all detection heads
        pred_distri = torch.cat([i.view(box_outputs[0].shape[0], self.reg_max * 4, -1) 
                                 for i in box_outputs], dim=2)
        pred_scores = torch.cat([i.view(cls_outputs[0].shape[0], self.nc, -1) 
                                for i in cls_outputs], dim=2)
        pred_poses = torch.cat([i.view(pose_outputs[0].shape[0], self.np, -1) 
                               for i in pose_outputs], dim=2)
        
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_poses = pred_poses.permute(0, 2, 1).contiguous()  # [B, anchors, num_poses]

        data_type = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        input_size = torch.tensor(box_outputs[0].shape[2:], device=self.device, 
                                 dtype=data_type) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(box_outputs, self.stride, offset=0.5)

        # Prepare targets
        idx = targets['idx'].view(-1, 1)
        cls = targets['cls'].view(-1, 1)
        box = targets['box']
        pose = targets['pose'].view(-1, 1) if 'pose' in targets else None

        # Combine targets
        if pose is not None:
            targets_combined = torch.cat((idx, cls, box, pose), dim=1).to(self.device)
        else:
            targets_combined = torch.cat((idx, cls, box), dim=1).to(self.device)
        
        if targets_combined.shape[0] == 0:
            gt = torch.zeros(batch_size, 0, 6 if pose is not None else 5, device=self.device)
        else:
            i = targets_combined[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            gt = torch.zeros(batch_size, counts.max(), 
                           6 if pose is not None else 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    gt[j, :n] = targets_combined[matches, 1:]
            
            # Convert bbox to xyxy format
            x = gt[..., 1:5].mul_(input_size[[1, 0, 1, 0]])
            y = torch.empty_like(x)
            dw = x[..., 2] / 2
            dh = x[..., 3] / 2
            y[..., 0] = x[..., 0] - dw
            y[..., 1] = x[..., 1] - dh
            y[..., 2] = x[..., 0] + dw
            y[..., 3] = x[..., 1] + dh
            gt[..., 1:5] = y
        
        # Split gt into labels, boxes, and poses
        if pose is not None:
            gt_labels, gt_bboxes, gt_poses = gt.split((1, 4, 1), 2)
        else:
            gt_labels, gt_bboxes = gt.split((1, 4), 2)
            gt_poses = None
        
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Decode predicted boxes
        pred_bboxes = self.box_decode(anchor_points, pred_distri)
        
        # Assign targets to anchors
        assigned_targets = self.assigner(pred_scores.detach().sigmoid(),
                                         (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                                         anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_bboxes, target_scores, fg_mask = assigned_targets

        target_scores_sum = max(target_scores.sum(), 1)

        # Classification loss
        loss_cls = self.cls_loss(pred_scores, target_scores.to(data_type)).sum() / target_scores_sum

        # Box and DFL losses
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss_box, loss_dfl = self.box_loss(pred_distri,
                                               pred_bboxes,
                                               anchor_points,
                                               target_bboxes,
                                               target_scores,
                                               target_scores_sum, fg_mask)

        # Pose classification loss
        loss_pose = torch.zeros(1, device=self.device)
        if gt_poses is not None and fg_mask.sum():
            # Get pose targets for matched anchors
            # We need to map gt_poses to the assigned anchors
            # For simplicity, we'll compute pose loss only on foreground anchors
            
            # Expand gt_poses to match anchor assignments
            batch_size, num_anchors, _ = pred_poses.shape
            
            # Create pose target tensor
            pose_targets = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=self.device)
            
            # For each batch, assign pose labels based on target assignment
            for b in range(batch_size):
                if mask_gt[b].sum() > 0:
                    # Get valid gt poses for this batch
                    # Convert mask_gt to bool for indexing
                    mask_bool = mask_gt[b].squeeze(-1).bool()
                    valid_poses = gt_poses[b][mask_bool].long().squeeze(-1)
                    
                    # Find which anchors are assigned to which gt boxes
                    # This is approximated using the foreground mask
                    fg_anchors = fg_mask[b]
                    
                    if fg_anchors.sum() > 0 and len(valid_poses) > 0:
                        # Simple assignment: distribute poses to foreground anchors
                        # In a more sophisticated implementation, you'd use the assigner's
                        # assignment indices
                        num_fg = fg_anchors.sum()
                        pose_idx = torch.arange(num_fg, device=self.device) % len(valid_poses)
                        pose_targets[b, fg_anchors] = valid_poses[pose_idx]
            
            # Compute cross-entropy loss only on foreground anchors
            if fg_mask.sum() > 0:
                fg_pred_poses = pred_poses[fg_mask]  # [num_fg, num_poses]
                fg_pose_targets = pose_targets[fg_mask]  # [num_fg]
                
                loss_pose = self.pose_loss(fg_pred_poses, fg_pose_targets).sum() / target_scores_sum

        # Apply loss weights
        loss_box *= self.params['box']
        loss_cls *= self.params['cls']
        loss_dfl *= self.params['dfl']
        loss_pose *= self.params.get('pose', 1.0)  # default weight = 1.0

        return loss_box, loss_cls, loss_dfl, loss_pose


# Test the loss function
if __name__ == '__main__':
    import yaml
    from utils.head_with_pose import HeadWithPose
    
    print("Testing ComputeLossWithPose...")
    
    # Create a simple model with HeadWithPose
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = HeadWithPose(version='n', num_classes=80, num_poses=3)
            self.head.stride = torch.tensor([8.0, 16.0, 32.0])
        
        def forward(self, x):
            return self.head(x)
    
    model = SimpleModel()
    
    params = {
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 1.0
    }
    
    criterion = ComputeLossWithPose(model, params, num_poses=3)
    
    # Create dummy outputs (training mode)
    model.head.train()
    x1 = torch.rand(2, 64, 80, 80)
    x2 = torch.rand(2, 128, 40, 40)
    x3 = torch.rand(2, 128, 20, 20)
    outputs = model.head([x1, x2, x3])
    
    # Create dummy targets
    targets = {
        'cls': torch.tensor([0, 0, 0]),  # person class
        'box': torch.rand(3, 4),
        'idx': torch.tensor([0, 0, 1]),
        'pose': torch.tensor([0, 2, 1])  # lying, standing, sitting
    }
    
    loss_box, loss_cls, loss_dfl, loss_pose = criterion(outputs, targets)
    
    print(f"\nLosses:")
    print(f"  Box loss: {loss_box.item():.4f}")
    print(f"  Cls loss: {loss_cls.item():.4f}")
    print(f"  DFL loss: {loss_dfl.item():.4f}")
    print(f"  Pose loss: {loss_pose.item():.4f}")
    print(f"  Total: {(loss_box + loss_cls + loss_dfl + loss_pose).item():.4f}")

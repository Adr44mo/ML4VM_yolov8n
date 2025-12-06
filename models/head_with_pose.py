"""
Modified YOLOv8 Head with Pose Detection
Adds a pose classification branch for detecting: lying, sitting, standing
"""

import torch
import torch.nn as nn


class Conv(nn.Module):
    """Standard convolution with BatchNorm and SiLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DFL(nn.Module):
    """Distribution Focal Loss"""
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(1, 2)
        x = x.softmax(1)
        x = self.conv(x)
        return x.view(b, 4, a)


def yolo_params(version):
    """Get depth, width, ratio parameters based on model version"""
    if version == 'n':
        return 1/3, 1/4, 2.0
    elif version == 's':
        return 1/3, 1/2, 2.0
    elif version == 'm':
        return 2/3, 3/4, 1.5
    elif version == 'l':
        return 1.0, 1.0, 1.0
    elif version == 'x':
        return 1.0, 1.25, 1.0


class HeadWithPose(nn.Module):
    """
    Modified YOLOv8 detection head with pose classification
    
    Outputs:
        - Bounding box coordinates (with DFL)
        - Object class scores (80 classes for COCO)
        - Pose class scores (3 classes: lying, sitting, standing)
    """
    
    def __init__(self, version, ch=16, num_classes=80, num_poses=4):
        super().__init__()
        self.ch = ch
        self.coordinates = self.ch * 4
        self.nc = num_classes
        self.np = num_poses  # number of pose classes
        self.no = self.coordinates + self.nc  # original output channels
        
        self.stride = torch.zeros(3)
        
        d, w, r = yolo_params(version=version)
        
        # Bounding box detection heads
        self.box = nn.ModuleList([
            nn.Sequential(
                Conv(int(256*w), self.coordinates, kernel_size=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(int(512*w), self.coordinates, kernel_size=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(int(512*w*r), self.coordinates, kernel_size=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)
            )
        ])
        
        # Object classification heads
        self.cls = nn.ModuleList([
            nn.Sequential(
                Conv(int(256*w), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(int(512*w), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(int(512*w*r), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)
            )
        ])
        
        # NEW: Pose classification heads
        self.pose = nn.ModuleList([
            nn.Sequential(
                Conv(int(256*w), self.np, kernel_size=3, stride=1, padding=1),
                Conv(self.np, self.np, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.np, self.np, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(int(512*w), self.np, kernel_size=3, stride=1, padding=1),
                Conv(self.np, self.np, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.np, self.np, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                Conv(int(512*w*r), self.np, kernel_size=3, stride=1, padding=1),
                Conv(self.np, self.np, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.np, self.np, kernel_size=1, stride=1)
            )
        ])
        
        # DFL for bbox refinement
        self.dfl = DFL()

    def forward(self, x):
        """
        Args:
            x: List of 3 feature maps from neck
        
        Returns:
            If training: list of 3 tensors [box, cls, pose] for each detection head
            If inference: concatenated predictions with refined boxes
        """
        box_outputs = []
        cls_outputs = []
        pose_outputs = []
        
        for i in range(len(self.box)):
            box = self.box[i](x[i])
            cls = self.cls[i](x[i])
            pose = self.pose[i](x[i])
            
            box_outputs.append(box)
            cls_outputs.append(cls)
            pose_outputs.append(pose)
        
        # In training mode, return raw predictions
        if self.training:
            return box_outputs, cls_outputs, pose_outputs
        
        # In inference mode, apply DFL and concatenate
        anchors, strides = (i.transpose(0, 1) for i in self.make_anchors(box_outputs, self.stride))
        
        # Concatenate predictions from all detection heads
        box_cat = torch.cat([i.view(x[0].shape[0], self.coordinates, -1) for i in box_outputs], dim=2)
        cls_cat = torch.cat([i.view(x[0].shape[0], self.nc, -1) for i in cls_outputs], dim=2)
        pose_cat = torch.cat([i.view(x[0].shape[0], self.np, -1) for i in pose_outputs], dim=2)
        
        # Apply DFL to refine boxes
        a, b = self.dfl(box_cat).chunk(2, 1)
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box_refined = torch.cat(tensors=((a + b) / 2, b - a), dim=1)
        
        # Return: [boxes, class_scores, pose_scores]
        return torch.cat(tensors=(box_refined * strides, cls_cat.sigmoid(), pose_cat.sigmoid()), dim=1)

    def make_anchors(self, x, strides, offset=0.5):
        """Generate anchor points for predictions"""
        assert x is not None
        anchor_tensor, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        
        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset
            sy, sx = torch.meshgrid(sy, sx)
            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        
        return torch.cat(anchor_tensor), torch.cat(stride_tensor)


# Test the modified head
if __name__ == '__main__':
    print("Testing HeadWithPose...")
    
    head = HeadWithPose(version='n', num_classes=80, num_poses=4)
    print(f"Parameters: {sum(p.numel() for p in head.parameters())/1e6:.3f}M")
    
    # Create dummy neck outputs
    x1 = torch.rand(2, 64, 80, 80)   # Detection head 1: 640/8 = 80
    x2 = torch.rand(2, 128, 40, 40)  # Detection head 2: 640/16 = 40
    x3 = torch.rand(2, 128, 20, 20)  # Detection head 3: 640/32 = 20
    
    # Test training mode
    head.train()
    box_out, cls_out, pose_out = head([x1, x2, x3])
    print(f"\nTraining mode outputs:")
    print(f"  Box outputs: {[b.shape for b in box_out]}")
    print(f"  Cls outputs: {[c.shape for c in cls_out]}")
    print(f"  Pose outputs: {[p.shape for p in pose_out]}")
    
    # Test inference mode
    head.eval()
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    output = head([x1, x2, x3])
    print(f"\nInference mode output: {output.shape}")
    print(f"  Channels: 4 (box) + 80 (cls) + 3 (pose) = {output.shape[1]}")

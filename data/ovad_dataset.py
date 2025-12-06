"""
OVAD Dataset loader for pose detection (sitting, standing, lying)
Modified from dataset.py to include pose annotations
"""

import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset


class OVADDataset(TorchDataset):
    """
    OVAD dataset loader for object detection with pose attributes
    Focuses on 'person' category with pose labels: lying, sitting, standing
    """
    
    def __init__(self, ovad_json_path, coco_img_dir, input_size=640, params=None, augment=False):
        """
        Args:
            ovad_json_path: Path to ovad2000.json file
            coco_img_dir: Directory containing COCO images (e.g., 'COCO/images/train2017')
            input_size: Input size for model (default 640)
            params: Training parameters
            augment: Whether to apply data augmentation
        """
        self.input_size = input_size
        self.params = params
        self.augment = augment
        self.coco_img_dir = coco_img_dir
        
        # Load OVAD annotations
        print(f"Loading OVAD dataset from {ovad_json_path}...")
        with open(ovad_json_path, 'r') as f:
            self.ovad_data = json.load(f)
        
        # Create mappings
        self.imgs = {img['id']: img for img in self.ovad_data['images']}
        
        # Filter annotations for 'person' category (id=1) with pose annotations
        # Pose attribute IDs: 94=lying, 95=sitting, 96=standing
        self.annotations = []
        for ann in self.ovad_data['annotations']:
            if ann['category_id'] == 1:  # person category
                pose_vec = ann['att_vec'][94:97]
                if any(v == 1 for v in pose_vec):  # has pose annotation
                    # Determine pose label
                    if ann['att_vec'][94] == 1:
                        pose_label = 0  # lying
                    elif ann['att_vec'][95] == 1:
                        pose_label = 1  # sitting
                    elif ann['att_vec'][96] == 1:
                        pose_label = 2  # standing
                    else:
                        continue
                    
                    ann['pose_label'] = pose_label
                    self.annotations.append(ann)
        
        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Get list of images with pose annotations
        self.image_ids = list(self.img_to_anns.keys())
        
        print(f"Loaded {len(self.image_ids)} images with {len(self.annotations)} person annotations with pose")
        
        # Pose class names
        self.pose_classes = ['lying', 'sitting', 'standing']
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        """
        Returns:
            image: (3, H, W) tensor
            targets: dict with keys:
                - 'cls': object class labels (all 0 for person)
                - 'box': bounding boxes in format (x_center, y_center, w, h) normalized
                - 'idx': batch index for each box
                - 'pose': pose labels (0=lying, 1=sitting, 2=standing)
        """
        img_id = self.image_ids[index]
        img_info = self.imgs[img_id]
        anns = self.img_to_anns[img_id]
        
        # Load image
        img_path = os.path.join(self.coco_img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]
        
        # Resize image
        r = self.input_size / max(h0, w0)
        if r != 1:
            image = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        h, w = image.shape[:2]
        
        # Pad to square
        dh, dw = self.input_size - h, self.input_size - w
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Normalize and transpose
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = np.ascontiguousarray(image)
        
        # Process annotations
        labels = []
        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            x, y, bw, bh = ann['bbox']
            
            # Scale bbox
            x = x * r + left
            y = y * r + top
            bw = bw * r
            bh = bh * r
            
            # Convert to center format and normalize
            x_center = (x + bw / 2) / self.input_size
            y_center = (y + bh / 2) / self.input_size
            w_norm = bw / self.input_size
            h_norm = bh / self.input_size
            
            # Label format: [class_id, x_center, y_center, w, h, pose_label]
            # class_id is 0 for person (we're only detecting one class)
            labels.append([0, x_center, y_center, w_norm, h_norm, ann['pose_label']])
        
        labels = np.array(labels) if len(labels) > 0 else np.zeros((0, 6))
        
        return torch.from_numpy(image), torch.from_numpy(labels)
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader
        Returns:
            images: (B, 3, H, W) tensor
            targets: dict with:
                - 'cls': (N,) tensor of class labels
                - 'box': (N, 4) tensor of boxes
                - 'idx': (N,) tensor of batch indices
                - 'pose': (N,) tensor of pose labels
        """
        images, labels = zip(*batch)
        
        # Stack images
        images = torch.stack(images, 0)
        
        # Process labels
        cls_list, box_list, idx_list, pose_list = [], [], [], []
        
        for i, lab in enumerate(labels):
            if lab.shape[0] > 0:
                cls_list.append(lab[:, 0])
                box_list.append(lab[:, 1:5])
                idx_list.append(torch.full((lab.shape[0],), i, dtype=torch.long))
                pose_list.append(lab[:, 5])
        
        if len(cls_list) > 0:
            targets = {
                'cls': torch.cat(cls_list, 0),
                'box': torch.cat(box_list, 0),
                'idx': torch.cat(idx_list, 0),
                'pose': torch.cat(pose_list, 0).long()
            }
        else:
            targets = {
                'cls': torch.zeros(0),
                'box': torch.zeros(0, 4),
                'idx': torch.zeros(0, dtype=torch.long),
                'pose': torch.zeros(0, dtype=torch.long)
            }
        
        return images, targets


# Simple test
if __name__ == '__main__':
    import yaml
    
    # Test the dataset
    ovad_json = '../ovad/ovad2000.json'
    img_dir = 'COCO/images/train2017'
    
    with open('args.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    dataset = OVADDataset(ovad_json, img_dir, input_size=640, params=params)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Pose classes: {dataset.pose_classes}")
    
    # Get a sample
    img, labels = dataset[0]
    print(f"\nSample image shape: {img.shape}")
    print(f"Sample labels shape: {labels.shape}")
    print(f"Sample labels:\n{labels}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=OVADDataset.collate_fn)
    
    batch = next(iter(loader))
    images, targets = batch
    print(f"\nBatch images shape: {images.shape}")
    print(f"Batch targets keys: {targets.keys()}")
    print(f"Number of objects in batch: {len(targets['cls'])}")
    print(f"Pose distribution: {torch.bincount(targets['pose'])}")

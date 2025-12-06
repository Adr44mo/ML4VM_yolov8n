"""
OVAD Dataset loader for pose detection
Uses OVAD annotations directly without COCO dependencies
"""

import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset


class OVADPoseDataset(TorchDataset):
    """
    OVAD dataset loader focusing on person detection with pose attributes
    Pose classes: lying (0), sitting (1), standing (2)
    
    This version uses OVAD annotations directly without requiring COCO annotation files.
    """
    
    def __init__(self, ovad_json_path, images_dir, input_size=640, augment=False, transform=None):
        """
        Args:
            ovad_json_path (str): Path to ovad2000.json file
            images_dir (str): Directory containing COCO images (e.g., 'data/COCO/images/val2017')
            input_size (int): Input size for model (default 640)
            augment (bool): Whether to apply data augmentation
            transform (callable): Optional transform to apply
        """
        self.input_size = input_size
        self.augment = augment
        self.transform = transform
        self.images_dir = Path(images_dir)
        
        # Pose configuration
        self.pose_classes = ['lying', 'sitting', 'standing']
        self.pose_attr_ids = {
            'lying': 94,      # position:horizontal/lying
            'sitting': 95,    # position:sitting/sit
            'standing': 96    # position:vertical/upright/standing
        }
        
        # Load OVAD annotations
        print(f"Loading OVAD dataset from {ovad_json_path}...")
        with open(ovad_json_path, 'r') as f:
            self.ovad_data = json.load(f)
        
        # Create image mapping
        self.imgs = {img['id']: img for img in self.ovad_data['images']}
        
        # Filter and process annotations
        self._process_annotations()
        
        print(f"✓ Loaded {len(self.image_ids)} images with {len(self.annotations)} person annotations")
        
        # Print pose distribution
        pose_counts = [0, 0, 0]
        for ann in self.annotations:
            pose_counts[ann['pose_label']] += 1
        print(f"  Pose distribution: lying={pose_counts[0]}, sitting={pose_counts[1]}, standing={pose_counts[2]}")
    
    def _process_annotations(self):
        """Process OVAD annotations to extract person instances with pose labels"""
        self.annotations = []
        
        for ann in self.ovad_data['annotations']:
            # Only process 'person' category (id=1)
            if ann['category_id'] != 1:
                continue
            
            # Check if annotation has pose attributes
            pose_vec = ann['att_vec'][94:97]  # Extract pose attributes
            
            if not any(v == 1 for v in pose_vec):
                continue  # Skip if no pose annotation
            
            # Determine pose label
            if ann['att_vec'][94] == 1:
                pose_label = 0  # lying
            elif ann['att_vec'][95] == 1:
                pose_label = 1  # sitting
            elif ann['att_vec'][96] == 1:
                pose_label = 2  # standing
            else:
                continue  # Should not happen, but skip if uncertain
            
            # Add pose label to annotation
            ann['pose_label'] = pose_label
            self.annotations.append(ann)
        
        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Get list of images with annotations
        self.image_ids = sorted(list(self.img_to_anns.keys()))
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        """
        Returns:
            image (Tensor): (3, H, W) normalized image tensor
            targets (dict): Dictionary with keys:
                - 'cls': class labels (Tensor) - all 0 for person
                - 'box': bounding boxes (Tensor) - normalized (x_center, y_center, w, h)
                - 'idx': batch indices (Tensor)
                - 'pose': pose labels (Tensor) - 0=lying, 1=sitting, 2=standing
        """
        img_id = self.image_ids[index]
        img_info = self.imgs[img_id]
        anns = self.img_to_anns[img_id]
        
        # Load image
        img_path = self.images_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Extract boxes and pose labels
        boxes = []
        pose_labels = []
        
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height] in absolute coordinates
            x, y, box_w, box_h = bbox
            
            # Convert to center format and normalize
            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            box_w_norm = box_w / w
            box_h_norm = box_h / h
            
            # Clip to valid range
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            box_w_norm = np.clip(box_w_norm, 0, 1)
            box_h_norm = np.clip(box_h_norm, 0, 1)
            
            boxes.append([x_center, y_center, box_w_norm, box_h_norm])
            pose_labels.append(ann['pose_label'])
        
        # Resize image to input size
        image_resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # Apply augmentation if enabled
        if self.augment:
            image_resized = self._augment(image_resized)
        
        # Apply custom transform if provided
        if self.transform:
            image_resized = self.transform(image_resized)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()  # HWC -> CHW
        
        # Create targets
        num_objects = len(boxes)
        targets = torch.zeros((num_objects, 6))  # [class, x_center, y_center, w, h, pose]
        
        for i, (box, pose) in enumerate(zip(boxes, pose_labels)):
            targets[i, 0] = 0  # class = 0 (person)
            targets[i, 1:5] = torch.tensor(box)
            targets[i, 5] = pose
        
        return image_tensor, targets
    
    def _augment(self, image):
        """Apply simple data augmentation"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            factor = 1.0 + np.random.uniform(-0.2, 0.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if np.random.rand() > 0.5:
            factor = 1.0 + np.random.uniform(-0.2, 0.2)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        return image
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader
        
        Args:
            batch: List of (image, targets) tuples
        
        Returns:
            images (Tensor): Batched images [B, 3, H, W]
            targets (dict): Dictionary with batched targets
        """
        images = []
        all_boxes = []
        all_classes = []
        all_poses = []
        all_indices = []
        
        for batch_idx, (image, targets) in enumerate(batch):
            images.append(image)
            
            if len(targets) > 0:
                all_classes.append(targets[:, 0])
                all_boxes.append(targets[:, 1:5])
                all_poses.append(targets[:, 5])
                all_indices.append(torch.full((len(targets),), batch_idx, dtype=torch.long))
        
        # Stack images
        images = torch.stack(images, dim=0)
        
        # Concatenate targets
        if len(all_classes) > 0:
            targets_dict = {
                'cls': torch.cat(all_classes, dim=0).long(),
                'box': torch.cat(all_boxes, dim=0),
                'pose': torch.cat(all_poses, dim=0).long(),
                'idx': torch.cat(all_indices, dim=0)
            }
        else:
            # Empty batch
            targets_dict = {
                'cls': torch.zeros(0, dtype=torch.long),
                'box': torch.zeros(0, 4),
                'pose': torch.zeros(0, dtype=torch.long),
                'idx': torch.zeros(0, dtype=torch.long)
            }
        
        return images, targets_dict
    
    def get_image_path(self, index):
        """Get the file path for an image"""
        img_id = self.image_ids[index]
        img_info = self.imgs[img_id]
        return self.images_dir / img_info['file_name']
    
    def get_annotations(self, index):
        """Get raw annotations for an image"""
        img_id = self.image_ids[index]
        return self.img_to_anns[img_id]


if __name__ == "__main__":
    # Test the dataset
    import sys
    sys.path.append('..')
    from configs.config import DataConfig
    
    dataset = OVADPoseDataset(
        ovad_json_path=DataConfig.OVAD_JSON_PATH,
        images_dir=DataConfig.COCO_IMG_DIR,
        input_size=640,
        augment=False
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Pose classes: {dataset.pose_classes}")
    
    # Get a sample
    image, targets = dataset[0]
    print(f"\nSample image shape: {image.shape}")
    print(f"Number of objects: {len(targets)}")
    print(f"Targets:\n{targets}")

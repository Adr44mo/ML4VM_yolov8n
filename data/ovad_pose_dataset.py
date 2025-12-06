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


# COCO category IDs are not contiguous, so we need a mapping to 0-79
COCO_CATEGORY_TO_IDX = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
    46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
    67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
    80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}

# Reverse mapping for decoding
IDX_TO_COCO_CATEGORY = {v: k for k, v in COCO_CATEGORY_TO_IDX.items()}

# COCO class names in order (0-79)
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class OVADPoseDataset(TorchDataset):
    """
    OVAD dataset loader for all COCO classes with pose attributes
    - Detects all 80 COCO object classes
    - Pose classes (4): lying (0), sitting (1), standing (2), other (3)
    - All objects get pose labels based on OVAD attributes if available
    - Objects without pose attributes get "other" (3)
    
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
        
        # Pose configuration (4 classes: lying, sitting, standing, other)
        self.pose_classes = ['lying', 'sitting', 'standing', 'other']
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
        
        print(f"✓ Loaded {len(self.image_ids)} images with {len(self.annotations)} annotations")
        
        # Print statistics
        category_counts = {}
        pose_counts = [0, 0, 0, 0]  # lying, sitting, standing, other
        for ann in self.annotations:
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
            if 'pose_label' in ann:
                pose_counts[ann['pose_label']] += 1
        print(f"  Categories: {len(category_counts)} unique classes")
        print(f"  Pose distribution: lying={pose_counts[0]}, sitting={pose_counts[1]}, standing={pose_counts[2]}, other={pose_counts[3]}")
    
    def _process_annotations(self):
        """Process OVAD annotations to extract all objects with pose labels"""
        self.annotations = []
        
        for ann in self.ovad_data['annotations']:
            # Check if annotation has pose attributes (lying, sitting, standing)
            pose_vec = ann['att_vec'][94:97]  # Extract pose attributes
            
            if any(v == 1 for v in pose_vec):
                # Determine pose label
                if ann['att_vec'][94] == 1:
                    pose_label = 0  # lying
                elif ann['att_vec'][95] == 1:
                    pose_label = 1  # sitting
                elif ann['att_vec'][96] == 1:
                    pose_label = 2  # standing
                else:
                    pose_label = 3  # other (shouldn't happen but safe default)
            else:
                # No specific pose annotation - use "other" class
                pose_label = 3  # other
            
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
                - 'cls': class labels (Tensor) - COCO class IDs (0-79)
                - 'box': bounding boxes (Tensor) - normalized (x_center, y_center, w, h)
                - 'idx': batch indices (Tensor)
                - 'pose': pose labels (Tensor) - 0=lying, 1=sitting, 2=standing, 3=other
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
        
        # Extract boxes, class labels, and pose labels
        boxes = []
        class_labels = []
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
            # COCO category_id is 1-indexed and non-contiguous, convert to 0-79 for model
            class_labels.append(COCO_CATEGORY_TO_IDX[ann['category_id']])
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
        
        for i, (box, cls, pose) in enumerate(zip(boxes, class_labels, pose_labels)):
            targets[i, 0] = cls  # class ID (0-79)
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

# YOLOv8 with Pose Detection

This repository extends YOLOv8 object detection to include **pose classification** (lying, sitting, standing) using the OVAD dataset.

## Overview

The original YOLOv8 model has been modified to perform multi-task learning:
1. **Object Detection**: Detect objects (80 COCO classes)
2. **Bounding Box Regression**: Localize objects with refined coordinates (DFL)
3. **Pose Classification**: Classify human poses into 3 categories

## Pose Classes

- **0: Lying/Horizontal** - Person in horizontal position
- **1: Sitting** - Person in seated position  
- **2: Standing/Vertical** - Person in upright position

## Architecture Modifications

### 1. HeadWithPose (`utils/head_with_pose.py`)

The detection head now has three parallel branches:

```
Neck Output → ┬─→ Box Head (bbox coordinates)
              ├─→ Class Head (80 object classes)
              └─→ Pose Head (3 pose classes) [NEW]
```

**Output channels:**
- Training: 3 separate outputs (box, cls, pose)
- Inference: 87 channels (4 bbox + 80 cls + 3 pose)

### 2. OVADDataset (`utils/ovad_dataset.py`)

Custom dataset loader for OVAD annotations:
- Filters for 'person' category (class_id=1)
- Extracts pose labels from attribute vectors:
  - `att_vec[94]`: lying
  - `att_vec[95]`: sitting
  - `att_vec[96]`: standing
- Returns targets with pose labels

### 3. ComputeLossWithPose (`utils/loss_with_pose.py`)

Extended loss computation with 4 components:
1. **Box Loss**: IoU-based bounding box regression loss
2. **Class Loss**: Binary cross-entropy for object classification
3. **DFL Loss**: Distribution focal loss for bbox refinement
4. **Pose Loss**: Cross-entropy for pose classification [NEW]

Total loss: `L = λ₁L_box + λ₂L_cls + λ₃L_dfl + λ₄L_pose`

## Dataset Setup

### OVAD Dataset Structure

```
ovad/
  ├── ovad2000.json          # Annotations with pose attributes
  └── README.md

COCO/
  ├── images/
  │   └── train2017/         # COCO images
  └── labels/
      └── train2017/         # YOLO format labels
```

### Statistics

From the OVAD dataset:
- **Total annotations**: ~10,000+
- **Person annotations with pose**: 3,915
  - Lying: 51 instances
  - Sitting: 1,075 instances
  - Standing: 2,789 instances

## Usage

### Training with Pose Detection

```python
from utils.head_with_pose import HeadWithPose
from utils.ovad_dataset import OVADDataset
from utils.loss_with_pose import ComputeLossWithPose

# Create model with pose head
class MyYoloWithPose(nn.Module):
    def __init__(self, version):
        super().__init__()
        self.backbone = Backbone(version=version)
        self.neck = Neck(version=version)
        self.head = HeadWithPose(version=version, num_classes=80, num_poses=3)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x[0], x[1], x[2])
        return self.head(list(x))

# Load OVAD dataset
dataset = OVADDataset(
    ovad_json_path='../ovad/ovad2000.json',
    coco_img_dir='COCO/images/train2017',
    input_size=640,
    params=params,
    augment=False
)

# Setup training
model = MyYoloWithPose(version='n').to(device)
model.head.stride = torch.tensor([8.0, 16.0, 32.0])

criterion = ComputeLossWithPose(model, params, num_poses=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    outputs = model(images)
    loss_box, loss_cls, loss_dfl, loss_pose = criterion(outputs, targets)
    total_loss = loss_box + loss_cls + loss_dfl + loss_pose
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

### Inference

```python
model.eval()
with torch.no_grad():
    output = model(image)
    # output shape: [batch, 87, num_anchors]
    
    boxes = output[:, :4, :]           # Bounding boxes
    class_scores = output[:, 4:84, :]  # Object classes
    pose_scores = output[:, 84:87, :]  # Pose classes
    
    # Get pose predictions
    pose_predictions = pose_scores.argmax(dim=1)
    # 0: lying, 1: sitting, 2: standing
```

## Model Comparison

| Model | Parameters | Output Channels | Tasks |
|-------|-----------|----------------|-------|
| YOLOv8n | 3.01M | 84 (4 + 80) | Detection only |
| YOLOv8n + Pose | ~3.02M | 87 (4 + 80 + 3) | Detection + Pose |

**Additional parameters**: ~10K (0.3% increase)

## Files Structure

```
yolov8_detection/
├── yolov8_from_scratch.ipynb    # Main notebook with pose extension
├── utils/
│   ├── ovad_dataset.py          # OVAD dataset loader
│   ├── head_with_pose.py        # Modified detection head
│   ├── loss_with_pose.py        # Extended loss computation
│   ├── dataset.py               # Original COCO dataset
│   └── util.py                  # Original utilities
├── POSE_DETECTION.md            # This file
└── README.md                    # Original README
```

## Training Tips

1. **Class Imbalance**: The dataset is heavily imbalanced (standing >> sitting >> lying)
   - Consider using weighted cross-entropy loss
   - Apply class-balanced sampling

2. **Memory Optimization**:
   - Use batch_size=2 for 8GB GPU
   - Disable augmentation initially
   - Set pin_memory=False

3. **Loss Weights**:
   - Default: `box=7.5, cls=0.5, dfl=1.5, pose=1.0`
   - Adjust `pose` weight if pose loss dominates

4. **Multi-task Learning**:
   - Pose classification helps learn better person representations
   - Joint training can improve detection accuracy

## Example Results

After training, the model can:
- Detect persons with bounding boxes
- Classify object categories (80 COCO classes)
- Predict pose state (lying/sitting/standing)

Example predictions:
```
Image 1:
  - Person @ (120, 80, 50, 150) - Standing (conf: 0.92)
  - Person @ (300, 200, 80, 60) - Sitting (conf: 0.87)

Image 2:
  - Person @ (50, 300, 200, 80) - Lying (conf: 0.95)
```

## References

1. **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
2. **OVAD**: [Open-vocabulary Attribute Detection](https://ovad-benchmark.github.io/)
3. **COCO Dataset**: [MS COCO](https://cocodataset.org/)

## Next Steps

1. **Improve Pose Assignment**: Current implementation uses simplified pose-to-anchor mapping. Consider using the assigner's matching indices for better accuracy.

2. **Handle Class Imbalance**: 
   - Implement focal loss for pose classification
   - Use data augmentation specific to underrepresented poses

3. **Extend to More Attributes**: OVAD has 117 attributes beyond pose (color, material, etc.)

4. **Post-processing**: Implement NMS considering both detection confidence and pose scores

5. **Evaluation**: Create metrics to evaluate pose classification accuracy alongside detection mAP

## License

Same as the original YOLOv8 implementation.

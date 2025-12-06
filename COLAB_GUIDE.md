# Using YOLOv8 from Google Colab

This guide shows you how to use this repository as a utility library for training YOLOv8 with pose detection on Google Colab.

## Quick Start

### Option 1: Use the Pre-made Colab Notebook

1. **Upload to Google Drive**:
   - Download `colab_train_pose.ipynb` from this repository
   - Upload it to your Google Drive
   - Open it with Google Colab

2. **Run the notebook**:
   - The notebook will automatically:
     - Clone this repository
     - Install dependencies
     - Download COCO and OVAD datasets
     - Train the model
     - Save checkpoints

### Option 2: Create Your Own Notebook

Create a new Colab notebook with the following structure:

```python
# 1. Clone the repository
!git clone https://github.com/Adr44mo/ML4VM_yolov8n.git
%cd ML4VM_yolov8n

# 2. Install dependencies
!pip install -q ultralytics opencv-python pyyaml matplotlib

# 3. Import model components
import sys
sys.path.append('utils')

from model_components import *
from head_with_pose import HeadWithPose
from ovad_dataset import OVADDataset
from loss_with_pose import ComputeLossWithPose

# 4. Create your model
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

# 5. Train!
model = MyYoloWithPose(version='n')
# ... your training code ...
```

## Repository Structure

```
ML4VM_yolov8n/
├── model_components.py          # All YOLOv8 building blocks (Conv, Backbone, Neck, Head)
├── utils/
│   ├── head_with_pose.py        # Pose detection head
│   ├── ovad_dataset.py          # OVAD dataset loader
│   ├── loss_with_pose.py        # Loss computation with pose
│   ├── util.py                  # Original YOLO utilities
│   ├── dataset.py               # Original COCO dataset
│   └── args.yaml                # Training parameters
├── colab_train_pose.ipynb       # Ready-to-use Colab notebook
├── yolov8_from_scratch.ipynb    # Local development notebook
└── README.md                    # This file
```

## What's Available

### Pre-built Components

1. **Model Architecture** (`model_components.py`):
   - `Conv`, `Bottleneck`, `C2f`, `SPPF` - Building blocks
   - `Backbone` - CSPDarknet53
   - `Neck` - Feature Pyramid Network
   - `Head` - Detection head with DFL
   - `MyYolo` - Complete YOLOv8 model

2. **Pose Detection** (`utils/`):
   - `HeadWithPose` - Extended head with pose classification
   - `OVADDataset` - Loads OVAD annotations with pose labels
   - `ComputeLossWithPose` - Multi-task loss (box + cls + dfl + pose)

### Model Versions

Choose from different YOLOv8 sizes:
- `'n'` - nano (3M params) - fastest
- `'s'` - small (11M params)
- `'m'` - medium (26M params)
- `'l'` - large (44M params)
- `'x'` - extra large (68M params)

## Datasets

### COCO Dataset
The notebook downloads COCO val2017 (~1GB) automatically.

**Or mount Google Drive if you already have it:**
```python
from google.colab import drive
drive.mount('/content/drive')
!ln -s /content/drive/MyDrive/datasets/COCO ./COCO
```

### OVAD Dataset
Download from: https://prior.allenai.org/projects/ovad

Upload `ovad2000.json` to the `ovad/` directory:
```python
from google.colab import files
uploaded = files.upload()
!mv ovad2000.json ovad/
```

## Training Configuration

Edit training parameters in `utils/args.yaml` or override in your notebook:

```python
import yaml

with open('utils/args.yaml') as f:
    params = yaml.safe_load(f)

# Modify parameters
params_pose = params.copy()
params_pose['pose'] = 1.0  # weight for pose loss
params_pose['box'] = 7.5   # weight for box loss
params_pose['cls'] = 0.5   # weight for classification loss
params_pose['dfl'] = 1.5   # weight for DFL loss
```

## Training Tips for Colab

### GPU Memory Management

```python
# Use smaller batch size for free Colab GPU (15GB)
batch_size = 4  # Increase if you have Colab Pro

# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Training loop with mixed precision
with autocast():
    outputs = model(images)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Save Checkpoints Regularly

```python
# Save to Google Drive
!mkdir -p /content/drive/MyDrive/yolo_checkpoints

checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, f'/content/drive/MyDrive/yolo_checkpoints/model_epoch_{epoch}.pt')
```

### Monitor Training

```python
# Install tensorboard
!pip install tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/pose_detection')

# In training loop
writer.add_scalar('Loss/total', total_loss, epoch)
writer.add_scalar('Loss/box', box_loss, epoch)
writer.add_scalar('Loss/pose', pose_loss, epoch)

# View in Colab
%tensorboard --logdir runs
```

## Example: Quick Training Script

```python
# Complete minimal training example
import torch
from torch.utils.data import DataLoader
import yaml

# Load config
with open('utils/args.yaml') as f:
    params = yaml.safe_load(f)

# Create dataset
dataset = OVADDataset(
    ovad_json_path='ovad/ovad2000.json',
    coco_img_dir='COCO/images/val2017',
    input_size=640,
    params=params,
    augment=True
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, 
                   collate_fn=OVADDataset.collate_fn)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyYoloWithPose('n').to(device)
model.head.stride = torch.tensor([8.0, 16.0, 32.0])

# Setup training
criterion = ComputeLossWithPose(model, params, num_poses=3)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Train
for epoch in range(10):
    for images, targets in loader:
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        outputs = model(images)
        loss_box, loss_cls, loss_dfl, loss_pose = criterion(outputs, targets)
        loss = loss_box + loss_cls + loss_dfl + loss_pose
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size`
- Use gradient accumulation
- Enable mixed precision training

### Dataset Not Found
- Check paths in `ovad_json_path` and `coco_img_dir`
- Make sure COCO images are in `COCO/images/val2017/`
- Verify `ovad2000.json` exists in `ovad/` directory

### Import Errors
```python
# Make sure utils is in path
import sys
sys.path.append('utils')

# Verify files exist
!ls utils/
```

## Additional Resources

- Original notebook: `yolov8_from_scratch.ipynb`
- YOLOv8 paper: https://arxiv.org/html/2304.00501v6
- OVAD dataset: https://prior.allenai.org/projects/ovad
- Ultralytics repo: https://github.com/ultralytics/ultralytics

## License

This implementation is for educational purposes. Please cite the original YOLOv8 paper if you use this code in your research.

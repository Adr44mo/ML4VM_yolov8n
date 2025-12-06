# YOLOv8 Detection Model with Pose Classification

Implementation of YOLOv8 detection model from scratch, extended with human pose detection (lying, sitting, standing) using the OVAD dataset.

## Features

- ✅ Complete YOLOv8 implementation from scratch (Backbone + Neck + Head)
- ✅ Extended with pose detection for human positions
- ✅ OVAD dataset integration for pose annotations
- ✅ Multi-task learning (object detection + pose classification)
- ✅ Google Colab ready for cloud training

## Quick Start

### Local Development
```bash
# Clone the repository
git clone https://github.com/Adr44mo/ML4VM_yolov8n.git
cd ML4VM_yolov8n

# Open the main notebook
jupyter notebook yolov8_from_scratch.ipynb
```

### Google Colab Training 🚀
**For training on Google Colab (recommended for GPU access):**

1. See **[COLAB_GUIDE.md](COLAB_GUIDE.md)** for detailed instructions
2. Use the ready-to-run notebook: **`colab_train_pose.ipynb`**
3. Or import components in your own notebook:
   ```python
   !git clone https://github.com/Adr44mo/ML4VM_yolov8n.git
   %cd ML4VM_yolov8n
   from model_components import *
   from utils.head_with_pose import HeadWithPose
   from utils.ovad_dataset import OVADDataset
   ```

## Repository Structure

```
├── model_components.py           # All YOLOv8 building blocks (standalone)
├── colab_train_pose.ipynb        # Google Colab training notebook
├── yolov8_from_scratch.ipynb     # Local development notebook
├── utils/
│   ├── head_with_pose.py         # Pose detection head
│   ├── ovad_dataset.py           # OVAD dataset loader
│   ├── loss_with_pose.py         # Multi-task loss
│   ├── dataset.py                # COCO dataset
│   ├── util.py                   # Training utilities
│   └── args.yaml                 # Training parameters
├── COLAB_GUIDE.md               # Complete Colab setup guide
├── POSE_DETECTION.md            # Pose detection details
└── README.md                    # This file
```

## Model Architecture

**Base YOLOv8:**
- Backbone: Modified CSPDarknet53
- Neck: Feature Pyramid Network
- Head: Detection head with DFL

**Pose Extension:**
- Additional pose classification branch (3 classes)
- Outputs: [bbox (4) + object class (80) + pose (3)] = 87 channels
- Trained on OVAD dataset annotations

## Pose Classes

- 🔴 **Lying** - horizontal/lying position
- 🔵 **Sitting** - sitting position
- 🟢 **Standing** - vertical/upright/standing position

## Training

### Local
```python
# See yolov8_from_scratch.ipynb
# Section 6: Pose Detection Extension
```

### Google Colab
```python
# See colab_train_pose.ipynb
# Or follow COLAB_GUIDE.md
```

## Model Versions

| Version | Parameters | Speed | Accuracy |
|---------|-----------|-------|----------|
| nano    | 3.0M      | ⚡⚡⚡  | ⭐⭐     |
| small   | 11M       | ⚡⚡    | ⭐⭐⭐   |
| medium  | 26M       | ⚡     | ⭐⭐⭐⭐ |

## References

### YOLOv8
1. https://github.com/jahongir7174/YOLOv8-dfl/tree/master 
2. https://arxiv.org/html/2304.00501v6/#S16
3. https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/yolo

### OVAD Dataset
4. https://prior.allenai.org/projects/ovad

### Tutorial
- YouTube: https://www.youtube.com/watch?v=6zQP0L-ph0M

## Citation

If you use this code in your research, please cite the original YOLOv8 paper:

```bibtex
@article{yolov8,
  title={YOLOv8: Real-Time Object Detection},
  author={Ultralytics},
  year={2023}
}
```

## License

Educational and research purposes. See individual file headers for specific licenses.



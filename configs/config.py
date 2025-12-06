"""
Configuration file for YOLOv8 Pose Detection Training
All paths, hyperparameters, and settings in one place
"""

import os
from pathlib import Path

# ===========================
# Project Structure
# ===========================

# Root directory (auto-detect)
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = ROOT_DIR / "data"
COCO_DIR = ROOT_DIR / "COCO"  # COCO is at root level
OVAD_DIR = ROOT_DIR / "ovad"  # OVAD is at root level

# COCO paths
COCO_IMAGES_DIR = COCO_DIR / "images" / "val2017"
COCO_LABELS_DIR = COCO_DIR / "labels" / "val2017"
COCO_ANNOTATIONS = COCO_DIR / "annotations" / "instances_val2017.json"

# OVAD paths
OVAD_JSON = OVAD_DIR / "ovad2000.json"

# Model directories
MODELS_DIR = ROOT_DIR / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Output directories
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
RESULTS_DIR = OUTPUTS_DIR / "results"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Create directories if they don't exist
for dir_path in [DATA_DIR, COCO_DIR, OVAD_DIR, MODELS_DIR, PRETRAINED_DIR, 
                 CHECKPOINTS_DIR, OUTPUTS_DIR, LOGS_DIR, RESULTS_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ===========================
# Model Configuration
# ===========================

class ModelConfig:
    """Model architecture configuration"""
    
    # Model version: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
    VERSION = 'n'
    
    # Number of classes (COCO)
    NUM_CLASSES = 80
    
    # Number of pose classes (OVAD)
    NUM_POSES = 3
    POSE_CLASSES = ['lying', 'sitting', 'standing']
    
    # Input image size
    INPUT_SIZE = 640
    
    # Detection head parameters
    DFL_CHANNELS = 16  # Distribution Focal Loss channels
    
    # Stride values for detection heads
    STRIDES = [8.0, 16.0, 32.0]
    
    # Pretrained weights (optional)
    PRETRAINED_WEIGHTS = PRETRAINED_DIR / "yolov8n.pt"
    USE_PRETRAINED = False  # Set to True to load pretrained backbone


# ===========================
# Dataset Configuration
# ===========================

class DataConfig:
    """Dataset and data loading configuration"""
    
    # OVAD dataset (primary source - has pose annotations)
    USE_OVAD = True
    OVAD_JSON_PATH = str(OVAD_JSON)
    COCO_IMG_DIR = str(COCO_IMAGES_DIR)
    
    # Data augmentation
    AUGMENT_TRAIN = True
    AUGMENT_VAL = False
    
    # Data loading
    BATCH_SIZE = 8  # Reduce if GPU memory is limited
    NUM_WORKERS = 4  # Number of data loading workers
    PIN_MEMORY = True
    SHUFFLE_TRAIN = True
    SHUFFLE_VAL = False
    
    # Train/Val split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Pose attribute IDs from OVAD
    POSE_ATTRIBUTES = {
        'lying': 94,      # position:horizontal/lying
        'sitting': 95,    # position:sitting/sit
        'standing': 96    # position:vertical/upright/standing
    }
    
    # Filter criteria
    PERSON_CATEGORY_ID = 1  # COCO person class ID


# ===========================
# Training Configuration
# ===========================

class TrainConfig:
    """Training hyperparameters"""
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Training schedule
    NUM_EPOCHS = 50
    WARMUP_EPOCHS = 3
    
    # Optimizer
    OPTIMIZER = 'AdamW'
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0005
    MOMENTUM = 0.937  # For SGD
    
    # Learning rate scheduler
    SCHEDULER = 'CosineAnnealingLR'  # Options: 'CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR'
    LR_MIN = 1e-5  # Minimum learning rate
    LR_DECAY_EPOCHS = 10  # For StepLR
    LR_DECAY_FACTOR = 0.1  # For StepLR
    
    # Loss weights
    LOSS_WEIGHTS = {
        'box': 7.5,      # Bounding box loss weight
        'cls': 0.5,      # Classification loss weight
        'dfl': 1.5,      # Distribution Focal Loss weight
        'pose': 1.0      # Pose classification loss weight
    }
    
    # Gradient clipping
    GRAD_CLIP = 10.0
    
    # Mixed precision training
    USE_AMP = True  # Automatic Mixed Precision
    
    # Gradient accumulation
    ACCUMULATE_STEPS = 1  # Accumulate gradients over N steps
    
    # Device
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    
    # Checkpoint saving
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    SAVE_BEST = True  # Save best model based on validation loss
    KEEP_LAST_N = 3  # Keep only last N checkpoints
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    TENSORBOARD = True
    WANDB = False  # Weights & Biases integration
    WANDB_PROJECT = 'yolov8-pose'
    
    # Resume training
    RESUME_CHECKPOINT = None  # Path to checkpoint to resume from


# ===========================
# Validation Configuration
# ===========================

class ValConfig:
    """Validation and evaluation configuration"""
    
    # Validation frequency
    VAL_INTERVAL = 1  # Validate every N epochs
    
    # Inference
    CONF_THRESHOLD = 0.25  # Confidence threshold for detections
    IOU_THRESHOLD = 0.45   # IoU threshold for NMS
    MAX_DETECTIONS = 300   # Maximum detections per image
    
    # Metrics to compute
    COMPUTE_MAP = True  # Compute mAP (mean Average Precision)
    COMPUTE_POSE_ACC = True  # Compute pose classification accuracy


# ===========================
# Visualization Configuration
# ===========================

class VisConfig:
    """Visualization settings"""
    
    # Colors for pose classes (RGB)
    POSE_COLORS = {
        'lying': (255, 0, 0),      # Red
        'sitting': (0, 0, 255),    # Blue
        'standing': (0, 255, 0)    # Green
    }
    
    # Visualization settings
    BOX_THICKNESS = 2
    TEXT_SIZE = 0.5
    TEXT_THICKNESS = 2
    
    # Save visualizations
    SAVE_VIS = True
    VIS_INTERVAL = 10  # Visualize every N batches during validation
    MAX_VIS_IMAGES = 10  # Maximum images to visualize per epoch


# ===========================
# Export Configuration
# ===========================

class ExportConfig:
    """Model export settings"""
    
    # Export formats
    EXPORT_ONNX = False
    EXPORT_TORCHSCRIPT = False
    
    # ONNX export settings
    ONNX_OPSET = 12
    ONNX_SIMPLIFY = True
    
    # TorchScript settings
    TORCHSCRIPT_OPTIMIZE = True


# ===========================
# Environment Configuration
# ===========================

class EnvConfig:
    """Environment and system settings"""
    
    # Random seeds for reproducibility
    SEED = 42
    
    # CUDA settings
    CUDA_BENCHMARK = True  # Enable cuDNN benchmark
    CUDA_DETERMINISTIC = False  # Deterministic operations (slower)
    
    # Number of threads
    NUM_THREADS = 4


# ===========================
# Helper Functions
# ===========================

def get_config_dict():
    """Get all configurations as a dictionary"""
    config = {
        'model': {k: v for k, v in ModelConfig.__dict__.items() if not k.startswith('_')},
        'data': {k: v for k, v in DataConfig.__dict__.items() if not k.startswith('_')},
        'train': {k: v for k, v in TrainConfig.__dict__.items() if not k.startswith('_')},
        'val': {k: v for k, v in ValConfig.__dict__.items() if not k.startswith('_')},
        'vis': {k: v for k, v in VisConfig.__dict__.items() if not k.startswith('_')},
        'export': {k: v for k, v in ExportConfig.__dict__.items() if not k.startswith('_')},
        'env': {k: v for k, v in EnvConfig.__dict__.items() if not k.startswith('_')},
    }
    return config


def print_config():
    """Print all configurations"""
    config = get_config_dict()
    print("=" * 80)
    print("YOLOv8 Pose Detection Configuration")
    print("=" * 80)
    for section, params in config.items():
        print(f"\n[{section.upper()}]")
        for key, value in params.items():
            print(f"  {key}: {value}")
    print("=" * 80)


def save_config(filepath):
    """Save configuration to a file"""
    import json
    from pathlib import Path
    
    config = get_config_dict()
    
    # Convert Path objects to strings
    def path_to_str(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: path_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [path_to_str(v) for v in obj]
        return obj
    
    config = path_to_str(config)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {filepath}")


# ===========================
# Quick Access
# ===========================

# Create instances for easy access
model_config = ModelConfig()
data_config = DataConfig()
train_config = TrainConfig()
val_config = ValConfig()
vis_config = VisConfig()
export_config = ExportConfig()
env_config = EnvConfig()


if __name__ == "__main__":
    # Print configuration when run directly
    print_config()
    
    # Save to file
    save_config(OUTPUTS_DIR / "config.json")

"""
YOLOv8 Multi-Class Object Detection with Pose Estimation Training Script for VSCode
This is a Python script version of the training notebook, designed for VSCode interactive mode.
Detects all 80 COCO object classes with pose estimation (lying/sitting/standing/other) for all objects.
Run sections with Shift+Enter or use Python Interactive window.
"""

# %%
# =============================================================================
# 1. IMPORTS AND SETUP
# =============================================================================

import sys
import torch
import torch.nn as nn
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Add project directories to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "models"))
sys.path.append(str(PROJECT_ROOT / "data"))
sys.path.append(str(PROJECT_ROOT / "configs"))

# Import configuration
from config import (
    ModelConfig, DataConfig, TrainConfig, ValConfig, VisConfig,
    model_config, data_config, train_config, val_config, vis_config,
    print_config
)

# Import model components
from model_components import (
    Conv, Bottleneck, C2f, SPPF, Upsample,
    Backbone, Neck, Head, DFL, MyYolo
)

# Import pose detection components
from head_with_pose import HeadWithPose
from loss_with_pose import ComputeLossWithPose
from ovad_pose_dataset import OVADPoseDataset

print("✓ All imports successful")
print_config()

# %%
# =============================================================================
# 2. DEVICE AND SEED SETUP
# =============================================================================

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set random seeds for reproducibility
torch.manual_seed(train_config.RANDOM_SEED)
np.random.seed(train_config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(train_config.RANDOM_SEED)

print(f"✓ Random seed set to {train_config.RANDOM_SEED}")

# %%
# =============================================================================
# 3. LOAD DATASET
# =============================================================================

print("\n" + "="*80)
print("LOADING DATASET")
print("="*80)

# Create dataset
dataset = OVADPoseDataset(
    ovad_json_path=data_config.OVAD_JSON_PATH,
    images_dir=data_config.COCO_IMG_DIR,
    input_size=model_config.INPUT_SIZE,
    augment=data_config.AUGMENT_TRAIN
)

print(f"\nDataset Info:")
print(f"  Total images: {len(dataset)}")
print(f"  Pose classes: {dataset.pose_classes}")
print(f"  Input size: {model_config.INPUT_SIZE}")

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=data_config.BATCH_SIZE,
    shuffle=data_config.SHUFFLE_TRAIN,
    num_workers=data_config.NUM_WORKERS,
    pin_memory=data_config.PIN_MEMORY,
    collate_fn=OVADPoseDataset.collate_fn
)

print(f"\nDataLoader Info:")
print(f"  Batch size: {data_config.BATCH_SIZE}")
print(f"  Number of batches: {len(dataloader)}")
print(f"  Total samples per epoch: {len(dataset)}")

# Get a sample batch
sample_images, sample_targets = next(iter(dataloader))
print(f"\nSample Batch:")
print(f"  Images shape: {sample_images.shape}")
print(f"  Number of objects: {len(sample_targets['cls'])}")
print(f"  Pose distribution: lying={sum(sample_targets['pose']==0)}, "
      f"sitting={sum(sample_targets['pose']==1)}, standing={sum(sample_targets['pose']==2)}")

# %%
# =============================================================================
# 4. CREATE MODEL
# =============================================================================

print("\n" + "="*80)
print("CREATING MODEL")
print("="*80)

class MyYoloWithPose(nn.Module):
    """YOLOv8 with Pose Detection Head"""
    def __init__(self, version='n'):
        super().__init__()
        self.backbone = Backbone(version=version)
        self.neck = Neck(version=version)
        self.head = HeadWithPose(
            version=version,
            num_classes=model_config.NUM_CLASSES,
            num_poses=model_config.NUM_POSES
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x[0], x[1], x[2])
        return self.head(list(x))

# Initialize model
model = MyYoloWithPose(version=model_config.VERSION).to(device)
model.head.stride = torch.tensor(model_config.STRIDES)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel: YOLOv8-{model_config.VERSION}")
print(f"  Total parameters: {total_params/1e6:.2f}M")
print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
print(f"  Backbone params: {sum(p.numel() for p in model.backbone.parameters())/1e6:.2f}M")
print(f"  Neck params: {sum(p.numel() for p in model.neck.parameters())/1e6:.2f}M")
print(f"  Head params: {sum(p.numel() for p in model.head.parameters())/1e6:.2f}M")

# %%
# =============================================================================
# 5. SETUP TRAINING
# =============================================================================

print("\n" + "="*80)
print("SETUP TRAINING")
print("="*80)

# Setup loss
loss_params = {
    'box': train_config.LOSS_WEIGHTS['box'],
    'cls': train_config.LOSS_WEIGHTS['cls'],
    'dfl': train_config.LOSS_WEIGHTS['dfl'],
    'pose': train_config.LOSS_WEIGHTS['pose']
}

criterion = ComputeLossWithPose(model, loss_params, num_poses=model_config.NUM_POSES)

# Setup optimizer
if train_config.OPTIMIZER == 'AdamW':
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.LEARNING_RATE,
        weight_decay=train_config.WEIGHT_DECAY
    )
elif train_config.OPTIMIZER == 'SGD':
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_config.LEARNING_RATE,
        momentum=train_config.MOMENTUM,
        weight_decay=train_config.WEIGHT_DECAY
    )

# Setup scheduler
if train_config.SCHEDULER == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config.NUM_EPOCHS,
        eta_min=train_config.LR_MIN
    )
elif train_config.SCHEDULER == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_config.LR_DECAY_EPOCHS,
        gamma=train_config.LR_DECAY_FACTOR
    )

# Setup AMP scaler for mixed precision
scaler = torch.cuda.amp.GradScaler() if train_config.USE_AMP else None

print(f"Optimizer: {train_config.OPTIMIZER}")
print(f"  Learning rate: {train_config.LEARNING_RATE}")
print(f"  Weight decay: {train_config.WEIGHT_DECAY}")
print(f"\nScheduler: {train_config.SCHEDULER}")
print(f"  Min LR: {train_config.LR_MIN}")
print(f"\nLoss weights:")
for key, val in loss_params.items():
    print(f"  {key}: {val}")
print(f"\nMixed precision training: {train_config.USE_AMP}")

# %%
# =============================================================================
# 6. TRAINING LOOP
# =============================================================================

print("\n" + "="*80)
print(f"STARTING TRAINING - {train_config.NUM_EPOCHS} EPOCHS")
print("="*80)

# Training history
history = {
    'epoch': [],
    'total_loss': [],
    'box_loss': [],
    'cls_loss': [],
    'dfl_loss': [],
    'pose_loss': [],
    'lr': []
}

# Create checkpoint directory
from configs.config import CHECKPOINTS_DIR
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

best_loss = float('inf')

model.train()
for epoch in range(train_config.NUM_EPOCHS):
    epoch_losses = {'total': 0, 'box': 0, 'cls': 0, 'dfl': 0, 'pose': 0}
    
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move to device
        images = images.float().to(device)
        targets['cls'] = targets['cls'].to(device)
        targets['box'] = targets['box'].to(device)
        targets['idx'] = targets['idx'].to(device)
        targets['pose'] = targets['pose'].to(device)
        
        # Forward pass with mixed precision
        if train_config.USE_AMP:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss_box, loss_cls, loss_dfl, loss_pose = criterion(outputs, targets)
                total_loss = loss_box + loss_cls + loss_dfl + loss_pose
        else:
            outputs = model(images)
            loss_box, loss_cls, loss_dfl, loss_pose = criterion(outputs, targets)
            total_loss = loss_box + loss_cls + loss_dfl + loss_pose
        
        # Backward pass
        optimizer.zero_grad()
        if train_config.USE_AMP:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.GRAD_CLIP)
            optimizer.step()
        
        # Accumulate losses
        epoch_losses['total'] += total_loss.item()
        epoch_losses['box'] += loss_box.item()
        epoch_losses['cls'] += loss_cls.item()
        epoch_losses['dfl'] += loss_dfl.item()
        epoch_losses['pose'] += loss_pose.item()
        
        # Log progress
        if (batch_idx + 1) % train_config.LOG_INTERVAL == 0:
            print(f"Epoch [{epoch+1}/{train_config.NUM_EPOCHS}] "
                  f"Batch [{batch_idx+1}/{len(dataloader)}] - "
                  f"Loss: {total_loss.item():.4f} "
                  f"(box: {loss_box.item():.4f}, cls: {loss_cls.item():.4f}, "
                  f"dfl: {loss_dfl.item():.4f}, pose: {loss_pose.item():.4f})")
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Calculate average losses
    num_batches = len(dataloader)
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
    
    # Save to history
    history['epoch'].append(epoch + 1)
    history['total_loss'].append(avg_losses['total'])
    history['box_loss'].append(avg_losses['box'])
    history['cls_loss'].append(avg_losses['cls'])
    history['dfl_loss'].append(avg_losses['dfl'])
    history['pose_loss'].append(avg_losses['pose'])
    history['lr'].append(current_lr)
    
    # Print epoch summary
    print(f"\n{'='*80}")
    print(f"Epoch [{epoch+1}/{train_config.NUM_EPOCHS}] Summary:")
    print(f"  Avg Total Loss: {avg_losses['total']:.4f}")
    print(f"  Box: {avg_losses['box']:.4f} | Cls: {avg_losses['cls']:.4f} | "
          f"DFL: {avg_losses['dfl']:.4f} | Pose: {avg_losses['pose']:.4f}")
    print(f"  Learning Rate: {current_lr:.6f}")
    print(f"{'='*80}\n")
    
    # Save checkpoint
    if (epoch + 1) % train_config.SAVE_INTERVAL == 0 or (epoch + 1) == train_config.NUM_EPOCHS:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'config': loss_params
        }
        checkpoint_path = CHECKPOINTS_DIR / f'yolov8n_pose_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if train_config.SAVE_BEST and avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            best_path = CHECKPOINTS_DIR / 'yolov8n_pose_best.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path} (loss: {best_loss:.4f})")
    print()

print("="*80)
print("TRAINING COMPLETE!")
print("="*80)

# %%
# =============================================================================
# 7. PLOT TRAINING CURVES
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Total loss
axes[0, 0].plot(history['epoch'], history['total_loss'], 'b-', linewidth=2)
axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

# Box loss
axes[0, 1].plot(history['epoch'], history['box_loss'], 'r-', linewidth=2)
axes[0, 1].set_title('Box Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True, alpha=0.3)

# Class loss
axes[0, 2].plot(history['epoch'], history['cls_loss'], 'g-', linewidth=2)
axes[0, 2].set_title('Classification Loss', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].grid(True, alpha=0.3)

# DFL loss
axes[1, 0].plot(history['epoch'], history['dfl_loss'], 'm-', linewidth=2)
axes[1, 0].set_title('DFL Loss', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True, alpha=0.3)

# Pose loss
axes[1, 1].plot(history['epoch'], history['pose_loss'], 'c-', linewidth=2)
axes[1, 1].set_title('Pose Loss', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].grid(True, alpha=0.3)

# Learning rate
axes[1, 2].plot(history['epoch'], history['lr'], 'orange', linewidth=2)
axes[1, 2].set_title('Learning Rate', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('LR')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
from configs.config import RESULTS_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(RESULTS_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Training curves saved to: {RESULTS_DIR / 'training_curves.png'}")

# %%
# =============================================================================
# 8. TEST INFERENCE
# =============================================================================

print("\n" + "="*80)
print("TESTING INFERENCE")
print("="*80)

model.eval()

# Get a test image
test_idx = 0
test_img_tensor, test_labels = dataset[test_idx]
test_img_display = test_img_tensor.numpy().transpose(1, 2, 0).astype(np.uint8)

# Run inference
test_img_batch = test_img_tensor.unsqueeze(0).float().to(device)
with torch.no_grad():
    predictions = model(test_img_batch)

print(f"Predictions shape: {predictions.shape}")
print(f"Expected: [1, 87, num_anchors] = [bbox(4) + classes(80) + poses(3)]")

# Extract predictions
pred_boxes = predictions[0, :4, :].cpu()
pred_class_scores = predictions[0, 4:84, :].cpu()
pred_pose_scores = predictions[0, 84:87, :].cpu()

# Filter for person detections
person_scores = pred_class_scores[0, :]
high_conf_mask = person_scores > val_config.CONF_THRESHOLD

filtered_boxes = pred_boxes[:, high_conf_mask]
filtered_poses = pred_pose_scores[:, high_conf_mask].argmax(dim=0)
filtered_scores = person_scores[high_conf_mask]

print(f"\nFound {filtered_boxes.shape[1]} high-confidence detections (threshold: {val_config.CONF_THRESHOLD})")

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(test_img_display)
ax.set_title(f'Pose Detection Results', fontsize=16)
ax.axis('off')

pose_names = model_config.POSE_CLASSES
pose_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]  # Red, Blue, Green

for i in range(filtered_boxes.shape[1]):
    x_c, y_c, w, h = filtered_boxes[:, i].numpy()
    pose_id = int(filtered_poses[i].item())
    score = filtered_scores[i].item()
    
    # Convert to corner format
    x1 = x_c - w/2
    y1 = y_c - h/2
    
    # Normalize color to [0, 1]
    color = tuple(c/255 for c in pose_colors[pose_id])
    
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                            edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1-5, f'{pose_names[pose_id]}: {score:.2f}', 
            color=color, fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
from configs.config import VISUALIZATIONS_DIR
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(VISUALIZATIONS_DIR / 'inference_result.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Visualization saved to: {VISUALIZATIONS_DIR / 'inference_result.png'}")

print("\n" + "="*80)
print("SCRIPT COMPLETE!")
print("="*80)

# %%

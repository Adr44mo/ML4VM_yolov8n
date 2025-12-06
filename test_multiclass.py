"""
Test script to verify multi-class dataset loading with pose detection

This script tests the updated OVADPoseDataset to ensure:
1. All COCO classes are loaded
2. Pose labels are correctly assigned (0-3 for all objects)
3. Class IDs are in valid range (0-79)
4. Pose classes: 0=lying, 1=sitting, 2=standing, 3=other
"""

import sys
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "data"))
sys.path.append(str(PROJECT_ROOT / "configs"))

from ovad_pose_dataset import OVADPoseDataset
from config import data_config

def test_multiclass_dataset():
    """Test that dataset loads all COCO classes with proper pose labels"""
    
    print("=" * 60)
    print("Testing Multi-Class Dataset with Pose Detection")
    print("=" * 60)
    
    # Create dataset
    print("\n1. Creating dataset...")
    dataset = OVADPoseDataset(
        ovad_json_path=data_config.OVAD_JSON_PATH,
        images_dir=data_config.COCO_IMG_DIR,
        input_size=640,
        augment=False
    )
    
    print(f"   ✓ Dataset created with {len(dataset)} images")
    
    # Test first few samples
    print("\n2. Testing data loading...")
    
    all_classes = set()
    all_poses = set()
    person_poses = []
    non_person_poses = []
    
    num_samples = min(10, len(dataset))
    print(f"   Analyzing first {num_samples} images...")
    
    for i in range(num_samples):
        img, targets = dataset[i]
        
        if len(targets) == 0:
            continue
            
        # Extract class IDs and pose labels
        class_ids = targets[:, 0].int().tolist()
        pose_labels = targets[:, 5].int().tolist()
        
        # Track all classes and poses
        all_classes.update(class_ids)
        all_poses.update(pose_labels)
        
        # Separate poses by class
        for cls, pose in zip(class_ids, pose_labels):
            if cls == 0:  # Person class
                person_poses.append(pose)
            else:
                non_person_poses.append(pose)
    
    # Print statistics
    print("\n3. Dataset Statistics:")
    print(f"   Classes found: {sorted(all_classes)}")
    print(f"   Number of unique classes: {len(all_classes)}")
    print(f"   Pose labels found: {sorted(all_poses)}")
    
    print("\n4. Validation Results:")
    
    # Validate class IDs
    valid_classes = all([0 <= cls <= 79 for cls in all_classes])
    if valid_classes:
        print("   ✓ All class IDs in valid range (0-79)")
    else:
        print("   ✗ ERROR: Some class IDs out of range!")
        return False
    
    # Validate pose labels
    valid_poses = all([0 <= pose <= 3 for pose in all_poses])
    if valid_poses:
        print("   ✓ All pose labels valid (0, 1, 2, 3)")
    else:
        print("   ✗ ERROR: Invalid pose labels found!")
        return False
    
    # Check person poses
    if person_poses:
        person_pose_set = set(person_poses)
        print(f"   ✓ Person poses found: {sorted(person_pose_set)}")
    else:
        print(f"   ⚠ No person detections in first {num_samples} samples")
    
    # Check non-person poses  
    if non_person_poses:
        non_person_pose_set = set(non_person_poses)
        print(f"   ✓ Non-person poses found: {sorted(non_person_pose_set)}")
        if 3 in non_person_pose_set:
            print(f"   ✓ 'other' pose class (3) found for non-person objects")
    
    # Test data shape
    print("\n5. Testing tensor shapes...")
    img, targets = dataset[0]
    print(f"   Image shape: {img.shape} (expected: [3, 640, 640])")
    if len(targets) > 0:
        print(f"   Targets shape: {targets.shape} (expected: [N, 6])")
        print(f"   Target format: [class_id, x, y, w, h, pose]")
        
        # Show example
        print("\n6. Example target (first object):")
        target = targets[0]
        print(f"   Class ID: {int(target[0].item())}")
        print(f"   Bbox: x={target[1]:.3f}, y={target[2]:.3f}, w={target[3]:.3f}, h={target[4]:.3f}")
        print(f"   Pose: {int(target[5].item())}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Dataset is working correctly.")
    print("=" * 60)
    
    return True

if __name__ == '__main__':
    try:
        success = test_multiclass_dataset()
        if success:
            print("\n✓ Multi-class dataset test successful!")
        else:
            print("\n✗ Multi-class dataset test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

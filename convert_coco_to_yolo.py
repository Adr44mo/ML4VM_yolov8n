import json
import os
from pathlib import Path

def convert_coco_to_yolo(json_file, output_dir, image_dir):
    """
    Convert COCO format annotations to YOLO format
    
    Args:
        json_file: Path to COCO JSON annotation file
        output_dir: Directory to save YOLO format labels
        image_dir: Directory containing the images (to verify image dimensions)
    """
    # Load COCO annotations
    print(f"Loading annotations from {json_file}...")
    with open(json_file, 'r') as f:
        coco = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create category ID mapping (COCO IDs are not sequential)
    # Map COCO category IDs to continuous 0-indexed class IDs
    categories = sorted(coco['categories'], key=lambda x: x['id'])
    coco_id_to_class_id = {cat['id']: idx for idx, cat in enumerate(categories)}
    print(f"Found {len(categories)} categories")
    
    # Create mappings
    images = {img['id']: img for img in coco['images']}
    
    # Group annotations by image_id
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    print(f"Converting {len(images)} images...")
    
    # Convert each image's annotations
    converted_count = 0
    for img_id, img_info in images.items():
        if converted_count % 1000 == 0:
            print(f"  Progress: {converted_count}/{len(images)}")
        
        # Get image dimensions
        img_width = img_info['width']
        img_height = img_info['height']
        img_filename = img_info['file_name']
        
        # Create label file path (same name as image but .txt extension)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        # Get annotations for this image
        anns = img_to_anns.get(img_id, [])
        
        # Convert annotations to YOLO format
        yolo_labels = []
        for ann in anns:
            # Get category_id and map it to 0-indexed class ID
            coco_category_id = ann['category_id']
            class_id = coco_id_to_class_id[coco_category_id]
            
            # Get bounding box [x, y, width, height] in COCO format (absolute pixels)
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Convert to YOLO format (normalized center coordinates)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            
            # Ensure values are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            norm_width = max(0, min(1, norm_width))
            norm_height = max(0, min(1, norm_height))
            
            # Format: <class_id> <x_center> <y_center> <width> <height>
            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        # Write labels to file
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))
        
        converted_count += 1
    
    print(f"✓ Converted {converted_count} images")
    print(f"✓ Labels saved to {output_dir}")


if __name__ == "__main__":
    # Paths
    coco_base = "../coco2017"
    yolo_base = "COCO"

    
    # Convert validation set
    print("\n=== Converting Validation Set ===")
    convert_coco_to_yolo(
        json_file=f"{coco_base}/annotations/instances_val2017.json",
        output_dir=f"{yolo_base}/labels/val2017",
        image_dir=f"{coco_base}/val2017"
    )
    
    print("\n✓ Conversion complete!")
    print(f"\nLabel structure:")
    print(f"  {yolo_base}/labels/train2017/  - Training labels")
    print(f"  {yolo_base}/labels/val2017/    - Validation labels")

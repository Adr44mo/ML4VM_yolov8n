import os

# Path to validation images folder
val_images_dir = 'COCO/images/val2017'

# Get all image files
image_files = []
if os.path.exists(val_images_dir):
    for filename in sorted(os.listdir(val_images_dir)):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            # Store full path relative to current directory
            image_files.append(os.path.join(val_images_dir, filename))

# Write to val2017.txt
output_file = 'COCO/val2017.txt'
with open(output_file, 'w') as f:
    for image_path in image_files:
        f.write(image_path + '\n')

print(f"Generated {output_file} with {len(image_files)} images")

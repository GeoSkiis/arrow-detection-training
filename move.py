import os
import shutil

def move_json_files(source_dir, target_dir_images, target_dir_jsons):
    if not os.path.exists(target_dir_images):
        os.makedirs(target_dir_images)
    if not os.path.exists(target_dir_jsons):
        os.makedirs(target_dir_jsons)

    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir_jsons, filename)
            shutil.move(source_path, target_path)
            print(f"Moved '{filename}' to '{target_dir_jsons}'")
        elif filename.endswith('.png'):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir_images, filename)
            shutil.move(source_path, target_path)
            print(f"Moved '{filename}' to '{target_dir_images}'")

# Usage
source_dir = 'raw_data'
target_dir_images = 'dataset/images'
target_dir_jsons = 'dataset/annotations'
move_json_files(source_dir, target_dir_images, target_dir_jsons)
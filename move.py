import os
import shutil

def move_json_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            shutil.move(source_path, target_path)
            print(f"Moved '{filename}' to '{target_dir}'")

# Usage
source_dir = 'dataset/images'
target_dir = 'dataset/annotations'
move_json_files(source_dir, target_dir)
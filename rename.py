import os

def rename_files(directory, start_number):
    files = os.listdir(directory)
    files.sort()  # Sort files to ensure consistent renaming order

    for index, filename in enumerate(files):
        new_name = f"{start_number + index}_image.png"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_name}'")

# Usage
directory = 'raw_data'
start_number = 1 # Change this to the desired starting number
rename_files(directory, start_number)
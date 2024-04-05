#move_pair_to_sub_folders
#give a base_path, divides folder image-caption.txt pairs into sub folders, each containing 2500 image-caption.txt pairs
#useful to divide image folder for caching

import os
import shutil

base_path = '/mnt/storage/projects/sdxl-train/pokemon/'

def move_image_caption_pairs(base_path):
    # Function to count items in a folder, returns 0 if folder does not exist
    def count_items_in_folder(folder_path):
        if not os.path.exists(folder_path):
            return 0
        else:
            return len(os.listdir(folder_path))
    
    # Find the starting point for sub_folder_count by checking existing sub-folders
    existing_sub_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.isdigit()]
    sub_folder_count = 0
    if existing_sub_folders:
        existing_sub_folders.sort()
        # Find the last sub-folder and start from the next one if it's full
        last_sub_folder = existing_sub_folders[-1]
        last_sub_folder_path = os.path.join(base_path, last_sub_folder)
        if count_items_in_folder(last_sub_folder_path) >= 5000:
            sub_folder_count = int(last_sub_folder) + 1
        else:
            # If the last folder is not full, continue using it
            sub_folder_count = int(last_sub_folder)
    
    # Initialize item_count for the current or new sub_folder
    sub_folder_path = os.path.join(base_path, f"{sub_folder_count:04d}")
    item_count = count_items_in_folder(sub_folder_path)
    max_items_per_folder = 5000  # Max items per sub-folder

    # List all files in the directory
    all_files = os.listdir(base_path)
    # Filter out the image files and their corresponding .txt files
    image_files = [f for f in all_files if f.split('.')[-1].lower() in ['jpg', 'webp', 'png', 'tif', 'gif']]
    # Sort to ensure matching pairs are processed together
    image_files.sort()

    for image_file in image_files:
        # Check if we need to create/move to a new sub-folder due to item count
        if item_count >= max_items_per_folder:
            sub_folder_count += 1
            sub_folder_path = os.path.join(base_path, f"{sub_folder_count:04d}")
            item_count = 0  # Reset item count for the new folder
        
        # Create the sub-folder if it doesn't exist
        if not os.path.exists(sub_folder_path):
            os.makedirs(sub_folder_path)

        # Define the base filename without extension to find the matching caption file
        base_filename = os.path.splitext(image_file)[0]
        caption_file = f"{base_filename}.txt"

        # Move both the image and caption file to the sub-folder
        shutil.move(os.path.join(base_path, image_file), os.path.join(sub_folder_path, image_file))
        shutil.move(os.path.join(base_path, caption_file), os.path.join(sub_folder_path, caption_file))
        
        # Update the item count (considering both files as a single item)
        item_count += 2

# Assuming the script is run in the directory with the image-caption pairs
# Replace 'your_directory_path' with the path to the directory containing your files
move_image_caption_pairs(base_path)

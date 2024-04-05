#searches directory for caption.txt files which do not have an accompanying image, and deletes them

import os

def delete_txt_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_path = os.path.join(root, file)
                img_path = os.path.splitext(txt_path)[0] + '.jpg'  # Assuming images have .jpg extension
                if not os.path.exists(img_path):
                    os.remove(txt_path)
                    print(f"Deleted {txt_path}")

# Example usage
directory_to_search = '/mnt/storage/projects/sdxl-train/pokemon/'
delete_txt_files(directory_to_search)
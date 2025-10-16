import os
import shutil

def create_yolo_structure(darkmark_path):
    # Define the new YOLOv8 folder structure
    h1 = 'valid'  # You can change this to 'train', 'valid', or 'test' as needed
    img_name = 'images'
    label_name = 'labels'
    yolo_structure = {
        h1: {
            img_name: 0, # Numnber used to count number of images and labels moved
            label_name: 0
        }
    }
    

    # Create the YOLOv8 folder structure
    for folder in yolo_structure:
        for subfolder in yolo_structure[folder]:
            os.makedirs(os.path.join(os.curdir, folder, subfolder), exist_ok=True)
    print(f"Created YOLOv8 folder structure in {darkmark_path}")

    # Move images and labels from Darkmark folder to the new structure
    root_dir_new_structure = os.path.join(os.curdir, h1)
    for file in os.listdir(darkmark_path):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy(os.path.join(darkmark_path, file), os.path.join(root_dir_new_structure, img_name, file))
            yolo_structure[h1][img_name] += 1
        elif file.endswith('.txt'):
            shutil.copy(os.path.join(darkmark_path, file), os.path.join(root_dir_new_structure, label_name, file))
            yolo_structure[h1][label_name] += 1
        elif file.endswith('.names'):
            shutil.copy(os.path.join(darkmark_path, file), os.path.join(root_dir_new_structure, file))  
            print(f"Moved {file} to {root_dir_new_structure}") 
    print(f"Moved {yolo_structure[h1][img_name]} images and {yolo_structure[h1][label_name]} labels to '{root_dir_new_structure}'.")

if __name__ == "__main__":
    darkmark_folder_path = input("Enter the path to the Darkmark folder to be converted (train/valid/test): ")
    create_yolo_structure(darkmark_folder_path)
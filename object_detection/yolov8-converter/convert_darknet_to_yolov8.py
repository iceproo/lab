import os
import shutil

def create_yolo_structure(darkmark_path):
    # Define the new YOLOv8 folder structure
    main_folder = os.path.basename(darkmark_path)

    image_name = "images"
    label_name = "labels"
    
    yolo_structure = {
        "train": {
            image_name: 0,
            label_name: 0
        },
        "valid": {
            image_name: 0,
            label_name: 0
        },
        "test": {
            image_name: 0,
            label_name: 0
        }
    }

    for folder_cat, content in yolo_structure.items():
        # Create the YOLOv8 folder structure
        for subfolder in content.keys():
            os.makedirs(os.path.join(main_folder, folder_cat, subfolder), exist_ok=True)
    print(f"Created YOLOv8 folder structure in {darkmark_path}")

    # Move images and labels from Darkmark folder to the new structure
    for folder_cat in yolo_structure.keys():
        if folder_cat not in os.listdir(darkmark_path):
            print(f"Warning: '{folder_cat}' folder not found in the Darkmark path.")
            continue
        root_dir_new_structure = os.path.join(main_folder, folder_cat)
        darkmark_folder_path = os.path.join(darkmark_path, folder_cat)

        for file in os.listdir(darkmark_folder_path):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(os.path.join(darkmark_folder_path, file), os.path.join(root_dir_new_structure, image_name, file))
                yolo_structure[folder_cat][image_name] += 1
            elif file.endswith('.txt'):
                shutil.copy(os.path.join(darkmark_folder_path, file), os.path.join(root_dir_new_structure, label_name, file))
                yolo_structure[folder_cat][label_name] += 1
            elif file.endswith('.names'):
                shutil.copy(os.path.join(darkmark_folder_path, file), os.path.join(root_dir_new_structure, file))  
                print(f"Moved {file} to {root_dir_new_structure}") 
        print(f"Moved {yolo_structure[folder_cat][image_name]} images and {yolo_structure[folder_cat][label_name]} labels to '{root_dir_new_structure}'.")

if __name__ == "__main__":
    darkmark_folder_path_main = input("Enter the path to the Darkmark folder to be converted (root): ")
    create_yolo_structure(darkmark_folder_path_main)
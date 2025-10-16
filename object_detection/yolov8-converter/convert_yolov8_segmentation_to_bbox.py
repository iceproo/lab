
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def seg_to_bbox(seg_str):
    # Example input: 2 0.207031 0.558594 0.208984 0.527344 0.210938 0.488281 0.214844 0.445312 0.21875 0.412109 0.222656 0.382812
    class_id, *points = seg_str.split()
    points = [float(p) for p in points]
    x_min, y_min, x_max, y_max = min(points[0::2]), min(points[1::2]), max(points[0::2]), max(points[1::2])
    # Calculate bbox center, width, height
    bw = x_max - x_min
    bh = y_max - y_min
    x_c = x_min + bw / 2
    y_c = y_min + bh / 2
    # Format: <class> <x_center> <y_center> <width> <height> 
    bbox_info = f"{class_id} {x_c} {y_c} {bw} {bh}"
    print(f"Segmentation: {seg_str}")
    print(f"Converted bbox: {bbox_info}")
    return bbox_info


# Added helper to visualize a single segmentation label on its image
def visualize_segmentation_on_image(image_path, label_path, path_to_save_img, color=(0, 255, 0), alpha=0.5):
    """
    Draws segmentation polygons from a YOLOv8 segmentation label file onto the image.
    Assumes each label line: <class_id> x1 y1 x2 y2 ... (normalized coordinates 0..1).
    If coordinates are absolute pixels, remove scaling by image size.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape[:2]
    overlay = img.copy()

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        toks = line.split()
        if len(toks) < 3:
            continue
        cls = toks[0]
        coords = [float(x) for x in toks[1:]]
        if len(coords) % 2 != 0:
            continue
        # convert normalized coords to pixels
        pts = []
        for x_norm, y_norm in zip(coords[0::2], coords[1::2]):
            x_px = int(round(x_norm * w))
            y_px = int(round(y_norm * h))
            pts.append((x_px, y_px))
        pts_np = np.array(pts, dtype=np.int32)
        if pts_np.size == 0:
            continue
        # fill polygon on overlay
        cv2.fillPoly(overlay, [pts_np], color)
        # outline
        cv2.polylines(img, [pts_np], isClosed=True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # put class id text
        cx, cy = pts_np[0]
        cv2.putText(img, str(cls), (cx, max(cy - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    # blend overlay
    #blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # convert BGR->RGB for matplotlib
    blended_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(blended_rgb)
    plt.axis("off")
    plt.savefig(path_to_save_img)
    plt.close()

    return img

def visualize_bboxes_on_img(image_path, label_path, img_save_path, color=(0, 255, 0)):
    """
    Draw bounding boxes from a YOLOv8-style label file onto an image and save the result.
    Label format per line: <class> <x_center> <y_center> <width> <height> [confidence]
    Coordinates are normalized (0..1).
    - class_names: optional list/dict to map class id -> name
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = image.shape[:2]

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        toks = line.split()
        if len(toks) < 5:
            continue
        cls_id = toks[0]
        try:
            x_c = float(toks[1]); y_c = float(toks[2])
            bw = float(toks[3]); bh = float(toks[4])
        except ValueError:
            continue

        print(f"Drawing box: {cls_id}, {x_c}, {y_c}, {bw}, {bh}")
        # convert normalized center->xyxy in pixels
        x1 = int(round((x_c - bw / 2) * w))
        y1 = int(round((y_c - bh / 2) * h))
        x2 = int(round((x_c + bw / 2) * w))
        y2 = int(round((y_c + bh / 2) * h))

        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)


        # draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)

        # build label text (class name and optional confidence)
        label_text = f"{cls_id} detected"

        # put text above box
        cv2.putText(image, label_text, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # convert BGR->RGB and save using matplotlib to preserve display quality
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.savefig(img_save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def get_sample_files(yolov8_segmentation_folder, index): 

    # print example of image and label file
    example_label = os.listdir(os.path.join(yolov8_segmentation_folder, "train", "labels"))[index]
    img_base_name = example_label.replace('.txt', '')
    img_file_name = img_base_name + '.jpg' # assuming images are in .jpg format

    img_file_to_print = os.path.join(yolov8_segmentation_folder, "train", "images", img_file_name)
    label_file_to_print = os.path.join(yolov8_segmentation_folder, "train", "labels", example_label)

    return img_file_to_print, label_file_to_print

def convert_yolov8_seg_to_bbox(yolov8_segmentation_folder):
    """
    Converts YOLOv8 segmentation labels to bounding box labels in place.
    Assumes folder structure:
    yolov8_segmentation_folder/
        train/
            images/
            labels/
        valid/
            images/
            labels/
        test/
            images/
            labels/
    Each label file in 'labels/' contains segmentation data to be converted to bounding boxes.

    Does this in place, overwriting original segmentation labels.   
    """
    for folder_img_category in ["train", "valid", "test"]:
        if not os.path.exists(os.path.join(yolov8_segmentation_folder, folder_img_category)):
            print(f"Folder '{folder_img_category}' does not exist in the provided path.")
            continue
        
        cur_dir = os.path.join(yolov8_segmentation_folder, folder_img_category, "labels")
        print(f"Processing labels in {cur_dir}")
        for file in os.listdir(cur_dir): 

            if file.endswith('.txt'):
                with open(os.path.join(cur_dir, file), 'r') as f:
                    seg_lines = f.readlines()

                bbox_lines = [seg_to_bbox(line.strip()) for line in seg_lines]

                with open(os.path.join(cur_dir, file), 'w') as f:
                    f.write("\n".join(bbox_lines))
                    print(f"Converted {file} to bounding box format.")

        print(f"Converted segmentation labels to bounding box labels in {cur_dir}")

if __name__ == "__main__":
    
    yolov8_segmentation_folder = input("Enter the path to the YOLOv8 segmentation folder: ")

    img_file_to_print, label_file_to_print = get_sample_files(yolov8_segmentation_folder, index=2)

    # Print segmentation mask on image
    visualize_segmentation_on_image(
        img_file_to_print,
        label_file_to_print, 
        "before_conversion_example.png"
    )

    convert_yolov8_seg_to_bbox(yolov8_segmentation_folder)
    
    # Print bbox on image
    visualize_bboxes_on_img(
        img_file_to_print,
        label_path=label_file_to_print, # Replace with actual model result if available
        img_save_path="after_conversion_example1.png"
    )
    img_file_to_print, label_file_to_print = get_sample_files(yolov8_segmentation_folder, index=2)
    print(f"Saved example images 'before_conversion_example.png' and 'after_conversion_example1.png' to visualize the conversion of file {img_file_to_print}.")
    visualize_bboxes_on_img(
        img_file_to_print,
        label_path=label_file_to_print, # Replace with actual model result if available
        img_save_path="after_conversion_example2.png"
    )
    print(f"Saved 'after_conversion_example2.png' based on image {img_file_to_print}.")

    print("Conversion complete.")
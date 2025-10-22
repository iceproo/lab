from typing import Tuple
import tempfile
import os

# /home/c21/c21ion/edu/exjobb/lab/object_detection/yolov8-converter/merge_bbox.py


def parse_darknet_line(line: str) -> Tuple[int, float, float, float, float]:
    """
    Parse a single Darknet-format line: "class x_center y_center width height".
    Returns (cls, x_center, y_center, width, height).
    Raises ValueError for malformed lines.
    """
    parts = line.strip().split()
    if not parts:
        raise ValueError("empty line")
    if len(parts) < 5:
        raise ValueError(f"expected 5 values, got {len(parts)}: {parts}")
    cls = int(parts[0])
    xc, yc, w, h = map(float, parts[1:5])
    return cls, xc, yc, w, h


def darknet_to_corners(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """
    Convert Darknet center-format (xc, yc, w, h) to corners (x1, y1, x2, y2).
    Coordinates are assumed to be relative (e.g. normalized to [0,1]) unless otherwise noted.
    """
    x1 = xc - w/2
    y1 = yc - h/2
    x2 = xc + w/2
    y2 = yc + h/2

    return x1, y1, x2, y2


# Convenience: parse a line and return class and corners
def parse_line_to_corners(line: str) -> Tuple[int, float, float, float, float]:
    cls, xc, yc, w, h = parse_darknet_line(line)
    x1, y1, x2, y2 = darknet_to_corners(xc, yc, w, h)
    return cls, x1, y1, x2, y2

def get_corners_merged_bbox_in_file(file_path:str) -> Tuple[int, float, float, float, float]:
    """
    Merges all bounding boxes in a Darknet-format annotation file into the single largest enclosing bounding box.
    Returns (class, x1, y1, x2, y2) of the merged bounding box.
    Assumes all boxes belong to the same class.
    """
    x1_min, y1_min = float('inf'), float('inf')
    x2_max, y2_max = float('-inf'), float('-inf')
    cls = None

    with open(file_path, 'r') as f:
        for line in f:
            line_cls, x1, y1, x2, y2 = parse_line_to_corners(line)
            if cls is None:
                cls = line_cls
            elif cls != line_cls:
                raise ValueError(f"Multiple classes found in file {file_path}")

            x1_min = min(x1_min, x1)
            y1_min = min(y1_min, y1)
            x2_max = max(x2_max, x2)
            y2_max = max(y2_max, y2)

    if cls is None:
        raise ValueError(f"No bounding boxes found in file {file_path}")

    return cls, x1_min, y1_min, x2_max, y2_max

def _run_merge_bbox_tests():
    # Create a temporary annotation file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        tmpfile.write("0 0.5 0.5 0.4 0.4\n")  # Box 1
        tmpfile.write("0 0.7 0.7 0.2 0.2\n")  # Box 2
        tmpfile.write("")  # Empty line to test robustness
        tmpfile.write("0 0.5 0.1 0.1 0.1\n") # Box 3, small box middle top
        tmpfile_path = tmpfile.name

    try:
        cls, x1, y1, x2, y2 = get_corners_merged_bbox_in_file(tmpfile_path)
        assert cls == 0
        # Box 1 corners: (0.3, 0.3) to (0.7, 0.7)
        # Box 2 corners: (0.6, 0.6) to (0.8, 0.8)
        # Box 3 corners: (0.45, 0.05) to (0.55, 0.15)
        # Merged box corners should be: (0.3, 0.05) to (0.8, 0.8)
        assert abs(x1 - 0.3) < 1e-8
        assert abs(y1 - 0.05) < 1e-8
        assert abs(x2 - 0.8) < 1e-8
        assert abs(y2 - 0.8) < 1e-8
    finally:
        os.remove(tmpfile_path)

    

# Simple tests
def _run_tests_get_corners():
    # 1) Full-image box
    line = "0 0.5 0.5 1.0 1.0\n"
    cls, x1, y1, x2, y2 = parse_line_to_corners(line)
    assert cls == 0
    assert (x1, y1, x2, y2) == (0.0, 0.0, 1.0, 1.0)

    # 2) Center-left small box
    line = "2 0.25 0.5 0.2 0.4"
    cls, x1, y1, x2, y2 = parse_line_to_corners(line)
    assert cls == 2
    # expected x1=0.15, x2=0.35, y1=0.3, y2=0.7
    assert abs(x1 - 0.15) < 1e-8
    assert abs(x2 - 0.35) < 1e-8
    assert abs(y1 - 0.3) < 1e-8
    assert abs(y2 - 0.7) < 1e-8


    # 4) parse_darknet_line errors
    try:
        parse_darknet_line("")  # should raise
        assert False, "expected ValueError for empty line"
    except ValueError:
        pass

    try:
        parse_darknet_line("0 0.5 0.5")  # too few values
        assert False, "expected ValueError for too few values"
    except ValueError:
        pass

    print("Test get corners passed.")


def corners_to_darknet(corners_tuple: Tuple[int, float, float, float, float]) -> Tuple[int, float, float, float, float]:
    """
    Convert corners-format (class, x1, y1, x2, y2) to Darknet center-format
    (class, xc, yc, w, h). Performs basic validation that x2 >= x1 and y2 >= y1.
    """
    cls, x1, y1, x2, y2 = corners_tuple
    if x2 < x1 or y2 < y1:
        raise ValueError(f"invalid corners: x2 < x1 or y2 < y1 ({x1},{y1},{x2},{y2})")
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return cls, xc, yc, w, h

def _run_tests_corners_to_darknet():
    # 1) Full-image box
    cls, xc, yc, w, h = corners_to_darknet((0, 0.0, 0.0, 1.0, 1.0))
    assert cls == 0
    assert abs(xc - 0.5) < 1e-8
    assert abs(yc - 0.5) < 1e-8
    assert abs(w - 1.0) < 1e-8
    assert abs(h - 1.0) < 1e-8

    # 2) Center-left small box (reverse of earlier test case)
    cls, xc, yc, w, h = corners_to_darknet((2, 0.15, 0.3, 0.35, 0.7))
    assert cls == 2
    assert abs(xc - 0.25) < 1e-8
    assert abs(yc - 0.5) < 1e-8
    assert abs(w - 0.2) < 1e-8
    assert abs(h - 0.4) < 1e-8

    # 4) Invalid corners should raise
    try:
        corners_to_darknet((0, 0.5, 0.5, 0.4, 0.4))
        assert False, "expected ValueError for x2 < x1"
    except ValueError:
        pass

    print("corners_to_darknet tests passed.")

def _run_test_round_trip_conversion():
    # 3) Round-trip: parse a Darknet line -> corners -> back to Darknet
    line = "0 0.5 0.5 0.4 0.4\n"
    corner_info = parse_line_to_corners(line)
    cls1, xc1, yc1, w1, h1 = corners_to_darknet(corner_info)
    # original center-format for the line
    assert cls1 == corner_info[0]
    assert abs(xc1 - 0.5) < 1e-8
    assert abs(yc1 - 0.5) < 1e-8
    assert abs(w1 - 0.4) < 1e-8
    assert abs(h1 - 0.4) < 1e-8

    print("Round-trip conversion test passed.")

def _run_test_file_rewrite():
    # Create a temporary annotation file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        tmpfile.write("0 0.5 0.5 0.4 0.4\n")  # Box 1
        tmpfile.write("0 0.7 0.7 0.2 0.2\n")  # Box 2
        tmpfile_path = tmpfile.name

    try:
        overwrite_file_merge_bbox(tmpfile_path)
        # Read back and verify
        with open(tmpfile_path, 'r') as f:
            line = f.readline()
            cls, xc, yc, w, h = parse_darknet_line(line)
            assert cls == 0
            assert abs(xc - 0.55) < 1e-8
            assert abs(yc - 0.55) < 1e-8
            assert abs(w - 0.5) < 1e-8
            assert abs(h - 0.5) < 1e-8

        print("File rewrite test passed.")
    finally:
        os.remove(tmpfile_path)

    # ---------- Test empty file handling --------------
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmpfile:
        tmpfile_path = tmpfile.name # Empty file test

    try:
        overwrite_file_merge_bbox(tmpfile_path)
        with open(tmpfile_path, 'r') as f:
            content = f.read()
            assert content.strip() == "", "Expected empty file to remain empty"
    finally:
        os.remove(tmpfile_path)



def overwrite_file_merge_bbox(file):
    
    # Expected is (0.40, 0.37, 0.53, 0.9) --> 0.465, 0.64, 0.13, 0.53
    with open(file, "r") as f:
        content = f.read()
        if not content.strip():
            print(f"File {file} is empty. Skipping.")
            return file
        
    bbox_corners = get_corners_merged_bbox_in_file(file)
    merged_bbox_darknet = corners_to_darknet(bbox_corners)

    with open(file, "w") as f:
        f.write(f"{merged_bbox_darknet[0]} {merged_bbox_darknet[1]} {merged_bbox_darknet[2]} {merged_bbox_darknet[3]} {merged_bbox_darknet[4]}\n")
    print(f"File {file} overwritten with merged bbox.")

    return file

if __name__ == "__main__":
    _run_tests_get_corners()
    _run_merge_bbox_tests()
    _run_tests_corners_to_darknet()
    _run_test_round_trip_conversion()
    _run_test_file_rewrite()

    #file = "/home/c21/c21ion/edu/exjobb/lab/object_detection/yolov8-converter/file_org_copy.txt" # File to get merged bbox from
    dataset_folder = "/home/c21/c21ion/edu/exjobb/lab/object_detection/yolonas/YOLO-detection-final-training-6-yolov8"
    
    for f in os.listdir(dataset_folder):
        print(f"Found folder: {f}")
        if f not in ["train", "valid", "test"]:
            continue

        folder_to_convert = os.path.join(dataset_folder, f, "labels")
        files_in_folder = os.listdir(folder_to_convert)
        for file in files_in_folder:
            file = os.path.join(folder_to_convert, file)
            file = overwrite_file_merge_bbox(file)
        print(f"All files (={len(files_in_folder)}) in {folder_to_convert} processed for merging bounding boxes.")
    
    print("All done.")
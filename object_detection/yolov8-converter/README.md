# YOLOv8 Converter

This project provides a Python script to convert images and labels from the Darkmark folder structure into the YOLOv8 format. It automates the process of creating the necessary folder structure and moving files to their respective locations.

## Project Structure

```
yolov8-converter
├── convert.py          # Main conversion logic
└── README.md           # Project documentation
```

## Installation

No special dependencies, should be just copy and run. 

## Usage

To use the converter, run the `convert.py` script. The script will create the YOLOv8 folder structure and copy the images and labels accordingly. Folder to copy images from is requested by the program once started.

Example:

```
python convert.py 
```

## Expected Folder Structure

After running the conversion, the expected folder structure for YOLOv8 will be:

```
train
├── images
|   ├── file1.jpg
|   └── file2.jpg
├── labels
|   ├── file1.txt
|   └── file2.txt
└── labels
```

The initial folder structure should be of darknet format which means something like this:
```
darkmark_folder
├── file1.jpg
├── file1.txt
├── file2.jgp
├── file2.txt
└── dataset.names

```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
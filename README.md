# Automatic License Plate Recognition System

This project implements an automatic license plate recognition system using computer vision and deep learning techniques. The system can detect vehicles, track them, and read their license plates from video input. It is specifically designed for Turkish license plates but can be adapted for other formats.

This project is based on and modified from:

- [automatic-number-plate-recognition-python-yolov8](https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8)
- [Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8)

## Key Modifications from Original Projects

- Replaced EasyOCR with PaddleOCR to achieve better OCR performance
- Adapted the system specifically for Turkish license plates (format: 2-digit city code + letters + numbers)
- Implemented smart plate correction using Levenshtein distance algorithm
- Improved license plate selection logic using frequency-based approach
- Added automatic image resizing for better OCR performance

## Models

The project uses two license plate detection models:

1. `license_plate_detector.pt`: Pre-trained model from the original repository
2. `custom_license_plate_detector.pt`: Custom model trained on Turkish license plates
   - Dataset: [Roboflow ANPR Dataset](https://universe.roboflow.com/berat-yumak-mkkon/anpr-yu4tw/dataset/4)
   - Dataset size: 1,711 images
   - Split: 78% train, 14% validation, 8% test
   - Base model: YOLOv11s (fine-tuned)
   - Classes: 1 (license plate)

## Features

- Vehicle detection using YOLO
- Vehicle tracking using SORT algorithm
- License plate detection using custom YOLO model
- License plate text recognition using PaddleOCR
- Smart license plate text correction for Turkish plates
- Missing frame interpolation for smooth tracking
- Real-time visualization with bounding boxes and plate numbers
- Results export to CSV format

## Project Structure

```
.
├── main.py                 # Main application entry point
├── utils/                  # Utility modules
│   ├── license_plate_processor.py  # License plate processing and OCR
│   ├── vehicle_tracker.py          # Vehicle tracking functionality
│   ├── data_writer.py             # Results export functionality
│   ├── data_interpolator.py       # Missing frame interpolation
│   └── visualizer.py             # Results visualization
├── models/                 # Model files
│   ├── custom_license_plate_detector.pt         # Custom license plate detection model
│   └── license_plate_detector.pt    # Pre-trained model from the original repository
├── sort/                   # SORT tracking algorithm
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download SORT tracking algorithm:
   ```bash
   git clone https://github.com/abewley/sort
   ```
4. Download required model files:
   - YOLO COCO model (yolo11n.pt)
   - Custom license plate detection model (custom_license_plate_detector.pt)

## Usage

1. Place your input video in the project directory
2. Update the video path in `main.py` if needed
3. Run the application:
   ```bash
   python main.py
   ```
4. The system will:
   - Process the video and detect vehicles
   - Track vehicles and detect license plates
   - Read and correct license plate numbers
   - Interpolate missing frames
   - Generate visualization
   - Save results to CSV files

## Output Files

- `test.csv`: Initial detection results
- `test_interpolated.csv`: Results with interpolated missing frames
- `out.mp4`: Visualization video with bounding boxes and plate numbers

## Module Descriptions

### LicensePlateProcessor

- Handles OCR using PaddleOCR
- Implements smart text correction for Turkish plates
- Validates plate format
- Processes and enhances license plate images

### VehicleTracker

- Matches license plates to vehicles
- Tracks vehicles across frames
- Handles vehicle detection and tracking

### DataWriter

- Exports detection results to CSV
- Formats data for analysis
- Handles file I/O operations

### DataInterpolator

- Interpolates missing frames
- Smooths tracking data
- Handles bounding box interpolation

### Visualizer

- Creates visualization video
- Draws bounding boxes and plate numbers
- Enhances visualization with plate crops

## Dependencies

- ultralytics: YOLO implementation
- opencv-python: Image processing
- numpy: Numerical operations
- paddleocr: OCR for license plate text recognition
- paddlepaddle: Deep learning framework for PaddleOCR
- scipy: Scientific computing (for interpolation)
- pandas: Data manipulation
- SORT: Simple Online and Realtime Tracking algorithm (from [abewley/sort](https://github.com/abewley/sort))

## Acknowledgments

This project is based on the work from:

1. [automatic-number-plate-recognition-python-yolov8](https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8) by Computer Vision Engineer
2. [Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8) by Muhammad Zeerak Khan

The original projects provided the foundation for vehicle detection and tracking, which we have enhanced with improved OCR capabilities and Turkish license plate-specific optimizations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

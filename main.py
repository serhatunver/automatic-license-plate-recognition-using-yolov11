from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from utils.license_plate_processor import LicensePlateProcessor
from utils.vehicle_tracker import VehicleTracker
from utils.data_writer import DataWriter
from utils.data_interpolator import DataInterpolator
from utils.visualizer import Visualizer
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import os
from typing import Dict, Any
import argparse

app = FastAPI(
    title="License Plate Recognition API",
    description="API for detecting and recognizing license plates in videos",
    version="1.0.0"
)

class LicensePlateRecognition:
    def __init__(self, video_path, coco_model_path, license_plate_model_path):
        """
        Initialize the License Plate Recognition system
        
        Args:
            video_path (str): Path to the input video
            coco_model_path (str): Path to the YOLO COCO model
            license_plate_model_path (str): Path to the license plate detection model
        """
        self.video_path = video_path
        self.coco_model = YOLO(coco_model_path)
        self.license_plate_detector = YOLO(license_plate_model_path)
        self.mot_tracker = Sort()
        self.license_plate_processor = LicensePlateProcessor()
        self.vehicle_tracker = VehicleTracker()
        self.data_writer = DataWriter()
        self.data_interpolator = DataInterpolator()
        
        # Vehicle classes to track (2: car, 3: motorcycle, 5: bus, 7: truck)
        self.vehicles = [2, 3, 5, 7]
        self.results = {}

    def process_frame(self, frame, frame_nmr):
        """
        Process a single frame for license plate detection and recognition
        
        Args:
            frame: The video frame to process
            frame_nmr: Frame number for tracking
        """
        self.results[frame_nmr] = {}
        
        # Detect vehicles
        detections = self.coco_model(frame)[0]
        vehicle_detections = []
        
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.vehicles:
                vehicle_detections.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = self.mot_tracker.update(np.asarray(vehicle_detections))

        # Detect and process license plates
        license_plates = self.license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            car_info = self.vehicle_tracker.get_car(license_plate, track_ids)
            if car_info is not None:
                xcar1, ycar1, xcar2, ycar2, car_id = car_info

                # Process license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_plate_text, license_plate_score = self.license_plate_processor.read_license_plate(license_plate_crop)

                if license_plate_text is not None:
                    self.results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_score
                        }
                    }

    def run(self):
        """
        Run the license plate recognition system on the video
        """
        # Process video frames
        cap = cv2.VideoCapture(self.video_path)
        frame_nmr = -1
        ret = True

        while ret:
            frame_nmr += 1
            ret, frame = cap.read()
            if ret:
                self.process_frame(frame, frame_nmr)

        cap.release()

        # Write initial results
        self.data_writer.write_results(self.results, './test.csv')

        # Interpolate missing data
        self.data_interpolator.process_file('./test.csv', './test_interpolated.csv')

        # Visualize results
        visualizer = Visualizer(self.video_path)
        visualizer.run()

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Process a video file for license plate recognition
    
    Args:
        file: The video file to process
        
    Returns:
        Dict containing processing results
    """
    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Initialize and run the license plate recognition system
        lpr = LicensePlateRecognition(
            video_path=temp_path,
            coco_model_path='yolo11n.pt',
            license_plate_model_path='./models/license_plate_detector.pt'
        )
        lpr.run()

        # Clean up the temporary file
        os.unlink(temp_path)

        return {
            "status": "success",
            "message": "Video processed successfully",
            "results_file": "test_interpolated.csv"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

def process_video_file(video_path: str):
    """
    Process a video file from command line
    
    Args:
        video_path: Path to the video file
    """
    lpr = LicensePlateRecognition(
        video_path=video_path,
        coco_model_path='yolo11n.pt',
        license_plate_model_path='./models/license_plate_detector.pt'
    )
    lpr.run()

def main():
    parser = argparse.ArgumentParser(description='License Plate Recognition System')
    parser.add_argument('--mode', choices=['api', 'cli'], default='api',
                      help='Run mode: api for FastAPI server, cli for command line processing')
    parser.add_argument('--video', type=str, help='Path to video file (required for cli mode)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host for API server')
    parser.add_argument('--port', type=int, default=8000, help='Port for API server')
    
    args = parser.parse_args()
    
    if args.mode == 'api':
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.mode == 'cli':
        if not args.video:
            parser.error("--video argument is required for cli mode")
        process_video_file(args.video)

if __name__ == "__main__":
    main()

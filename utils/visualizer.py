import ast
import cv2
import numpy as np
import pandas as pd
from collections import Counter

class Visualizer:
    def __init__(self, video_path):
        """
        Initialize the visualizer
        
        Args:
            video_path (str): Path to the input video
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('./out.mp4', fourcc, self.fps, (self.width, self.height))
        
        # Load results
        self.results = pd.read_csv('./test_interpolated.csv')
        self.license_plate = {}
        self._process_license_plates()

    def _process_license_plates(self):
        """
        Process license plates for all cars
        """
        for car_id in np.unique(self.results['car_id']):
            car_results = self.results[self.results['car_id'] == car_id]
            possible_plates = car_results[['license_number', 'license_number_score', 'frame_nmr', 'license_plate_bbox']].to_dict('records')
            plate_number, plate_crop = self._select_best_plate(possible_plates)
            self.license_plate[car_id] = {
                'license_crop': plate_crop,
                'license_plate_number': plate_number
            }

    def _select_best_plate(self, possible_plates, top_n=10):
        """
        Select the best license plate from possible readings
        
        Args:
            possible_plates (list): List of possible plate readings
            top_n (int): Number of top plates to consider
            
        Returns:
            tuple: (plate_number, plate_crop) or (None, None)
        """
        if not possible_plates:
            return None, None

        # Filter valid plates
        valid_plates = [plate for plate in possible_plates if plate['license_number'] != '0' and plate['license_number'].strip()]

        if not valid_plates:
            return None, None

        # Sort by score and get top N
        sorted_plates = sorted(valid_plates, key=lambda x: x['license_number_score'], reverse=True)
        top_plates = sorted_plates[:min(top_n, len(sorted_plates))]

        # Find most common plate number
        license_numbers = [plate['license_number'] for plate in top_plates]
        if license_numbers:
            most_common_license_number = Counter(license_numbers).most_common(1)[0][0]
        else:
            return None, None

        # Find best result
        best_result = next((plate for plate in top_plates if plate['license_number'] == most_common_license_number), None)

        license_crop = None
        if best_result:
            frame_nmr = best_result['frame_nmr']
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
            ret, frame = self.cap.read()
            if ret:
                x1, y1, x2, y2 = ast.literal_eval(
                    best_result['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
        else:
            license_crop = np.zeros((400, 400, 3), dtype=np.uint8)

        return most_common_license_number, license_crop

    def draw_border(self, img, top_left, bottom_right, color=(0, 255, 0), thickness=5, line_length_x=200, line_length_y=200):
        """
        Draw a border around a region
        
        Args:
            img: Image to draw on
            top_left: Top-left corner coordinates
            bottom_right: Bottom-right corner coordinates
            color: Border color
            thickness: Line thickness
            line_length_x: Length of horizontal lines
            line_length_y: Length of vertical lines
            
        Returns:
            Image with border drawn
        """
        x1, y1 = top_left
        x2, y2 = bottom_right

        # Draw corner lines
        cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # Top-left vertical
        cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)  # Top-left horizontal

        cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # Bottom-left vertical
        cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)  # Bottom-left horizontal

        cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # Top-right horizontal
        cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)  # Top-right vertical

        cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # Bottom-right vertical
        cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)  # Bottom-right horizontal

        return img

    def process_frame(self, frame, frame_nmr):
        """
        Process a single frame for visualization
        
        Args:
            frame: Frame to process
            frame_nmr: Frame number
        """
        df_ = self.results[self.results['frame_nmr'] == frame_nmr]
        
        for row_indx in range(len(df_)):
            try:
                # Draw car border
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                    df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                self.draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 5,
                               line_length_x=200, line_length_y=200)

                # Draw license plate rectangle
                x1, y1, x2, y2 = ast.literal_eval(
                    df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)

                # Add license plate crop and text
                license_crop = self.license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
                H, W, _ = license_crop.shape

                # Resize license plate crop
                scale = 0.4
                license_crop_resized = cv2.resize(license_crop, (0, 0), fx=scale, fy=scale)
                H, W, _ = license_crop_resized.shape

                # Place license plate image
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                     int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop_resized

                # Add plate number text
                plate_number = self.license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2.0
                font_thickness = 6
                (text_width, text_height), _ = cv2.getTextSize(plate_number, font, font_scale, font_thickness)

                text_x = int((car_x2 + car_x1 - text_width) / 2)
                text_y = int(car_y1 - H - 100)

                # Draw text background
                cv2.rectangle(frame,
                            (text_x, text_y - text_height),
                            (text_x + text_width, text_y + int(text_height * 0.3)),
                            (255, 255, 255),
                            thickness=-1)

                # Draw text
                cv2.putText(frame,
                          plate_number,
                          (text_x, text_y),
                          font,
                          font_scale,
                          (0, 0, 0),
                          font_thickness)

            except Exception as e:
                print(f"Error processing frame {frame_nmr}: {e}")

    def run(self):
        """
        Run the visualization process
        """
        frame_nmr = -1
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret = True

        while ret:
            ret, frame = self.cap.read()
            frame_nmr += 1
            if ret:
                self.process_frame(frame, frame_nmr)
                self.out.write(frame)
                frame = cv2.resize(frame, (1280, 720))

        self.out.release()
        self.cap.release() 
class VehicleTracker:
    def __init__(self):
        """
        Initialize the vehicle tracker
        """
        pass

    def get_car(self, license_plate, vehicle_track_ids):
        """
        Match a license plate to a vehicle in the tracking data
        
        Args:
            license_plate: License plate detection data (x1, y1, x2, y2, score, class_id)
            vehicle_track_ids: List of tracked vehicles with their bounding boxes
            
        Returns:
            tuple: (x1, y1, x2, y2, car_id) if match found, None otherwise
        """
        x1, y1, x2, y2, score, class_id = license_plate
        
        for vehicle in vehicle_track_ids:
            xcar1, ycar1, xcar2, ycar2, car_id = vehicle
            
            # Check if license plate is inside vehicle bounding box
            if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                return vehicle
                
        return None 
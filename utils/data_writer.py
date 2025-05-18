class DataWriter:
    def __init__(self):
        """
        Initialize the data writer
        """
        pass

    def write_results(self, results, output_path):
        """
        Write detection results to a CSV file
        
        Args:
            results (dict): Dictionary containing detection results
            output_path (str): Path to output CSV file
        """
        with open(output_path, 'w') as f:
            # Write header
            f.write('{},{},{},{},{},{},{}\n'.format(
                'frame_nmr', 'car_id', 'car_bbox',
                'license_plate_bbox', 'license_plate_bbox_score',
                'license_number', 'license_number_score'
            ))
            
            # Write data
            for frame_nmr in results.keys():
                for car_id in results[frame_nmr].keys():
                    if ('car' in results[frame_nmr][car_id].keys() and
                        'license_plate' in results[frame_nmr][car_id].keys() and
                        'text' in results[frame_nmr][car_id]['license_plate'].keys()):
                        
                        # Format bounding boxes
                        car_bbox = '[{} {} {} {}]'.format(
                            results[frame_nmr][car_id]['car']['bbox'][0],
                            results[frame_nmr][car_id]['car']['bbox'][1],
                            results[frame_nmr][car_id]['car']['bbox'][2],
                            results[frame_nmr][car_id]['car']['bbox'][3]
                        )
                        
                        license_plate_bbox = '[{} {} {} {}]'.format(
                            results[frame_nmr][car_id]['license_plate']['bbox'][0],
                            results[frame_nmr][car_id]['license_plate']['bbox'][1],
                            results[frame_nmr][car_id]['license_plate']['bbox'][2],
                            results[frame_nmr][car_id]['license_plate']['bbox'][3]
                        )
                        
                        # Write row
                        f.write('{},{},{},{},{},{},{}\n'.format(
                            frame_nmr,
                            car_id,
                            car_bbox,
                            license_plate_bbox,
                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                            results[frame_nmr][car_id]['license_plate']['text'],
                            results[frame_nmr][car_id]['license_plate']['text_score']
                        )) 
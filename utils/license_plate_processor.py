from paddleocr import PaddleOCR
from itertools import product
import re
import cv2

class LicensePlateProcessor:
    def __init__(self):
        """
        Initialize the license plate processor with OCR and character mapping
        """
        # Initialize the PaddleOCR reader
        self.ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=True)
        
        # Character mapping for OCR correction
        self.dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
        
        # Advanced OCR error correction dictionary
        self.ocr_alternatives = {
            'O': ['D', '0', 'O'],
            'I': ['1', 'I'],
            'l': ['1', 'l', '1'],
            'J': ['3', 'J'],
            'A': ['4', 'A'],
            'G': ['6', 'G'],
            'S': ['5', 'S'],
            'B': ['8', 'B'],
            'Z': ['2', 'Z'],
            '0': ['D', '0', 'O'],
            '1': ['1', 'I', 'l'],
            '3': ['3', 'J'],
            '4': ['4', 'A'],
            '6': ['6', 'G'],
            '5': ['5', 'S'],
            '8': ['8', 'B'],
            '2': ['2', 'Z'],
            'D': ['O', 'D'],
            'T': ['T', '1'],
            'N': ['N'],
            'K': ['K'],
            'R': ['R'],
            'V': ['V']
        }

    def check_plate_format(self, plate):
        """
        Check if the plate matches Turkish license plate format (without spaces)
        
        Args:
            plate (str): License plate text to check
            
        Returns:
            bool: True if format is valid, False otherwise
        """
        pattern = r"^\d{2}[A-Z]{1,3}\d{2,4}$"
        return re.match(pattern, plate) is not None

    def generate_possible_plates(self, raw_plate):
        """
        Generate possible plate variations based on OCR errors
        
        Args:
            raw_plate (str): Raw plate text from OCR
            
        Yields:
            str: Possible plate variations
        """
        alternatives_per_char = [self.ocr_alternatives.get(char.upper(), [char.upper()]) for char in raw_plate]
        for combination in product(*alternatives_per_char):
            yield "".join(combination)

    def calculate_edit_distance(self, s1, s2):
        """
        Calculate Levenshtein distance between two strings
        
        Args:
            s1 (str): First string
            s2 (str): Second string
            
        Returns:
            int: Edit distance between strings
        """
        if len(s1) < len(s2):
            return self.calculate_edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1.upper() != c2.upper())
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def try_correct_plate_smart(self, raw_plate):
        """
        Try to correct plate text using all possible variations
        
        Args:
            raw_plate (str): Raw plate text to correct
            
        Returns:
            str: Corrected plate text
        """
        valid_plates = []
        for possible_plate in self.generate_possible_plates(raw_plate):
            if self.check_plate_format(possible_plate):
                valid_plates.append(possible_plate)

        if not valid_plates:
            return raw_plate

        best_plate = min(valid_plates, key=lambda plate: self.calculate_edit_distance(raw_plate.upper(), plate))
        return best_plate

    def license_complies_format_flexible(self, text):
        """
        Check if text complies with Turkish license plate format (more flexible)
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if format is valid, False otherwise
        """
        text = text.replace(" ", "")
        if not text or not text[:2].isdigit():
            return False
        rest = text[2:]
        l = len(rest)
        return (l >= 5 and l <= 7 and
                ((l == 5 and ((rest[0].isalpha() and rest[1:].isdigit()) or
                               (rest[:2].isalpha() and rest[2:].isdigit()) or
                               (rest[:3].isalpha() and rest[3:].isdigit()))) or
                 (l == 6 and ((rest[:2].isalpha() and rest[2:].isdigit()) or
                               (rest[:3].isalpha() and rest[3:].isdigit()))) or
                 (l == 7 and (rest[:3].isalpha() and rest[3:].isdigit()))))

    def format_license_paddle(self, text):
        """
        Format text from PaddleOCR output
        
        Args:
            text (str): Raw text from PaddleOCR
            
        Returns:
            str: Formatted text
        """
        text = text.upper().replace(" ", "")
        corrected = ''
        for i, char in enumerate(text):
            if i < 2 and char in self.dict_char_to_int:
                corrected += self.dict_char_to_int[char]
            else:
                corrected += char
        return corrected

    def read_license_plate(self, license_plate_crop):
        """
        Read license plate text from cropped image
        
        Args:
            license_plate_crop: Cropped license plate image
            
        Returns:
            tuple: (license_plate_text, confidence_score) or (None, None) if no valid plate found
        """
        # Resize image for better OCR
        target_height = 100
        scale = target_height / license_plate_crop.shape[0]
        resized = cv2.resize(license_plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        # Perform OCR
        result = self.ocr.ocr(resized, cls=True)
        
        if not result or not result[0]:
            return None, None

        texts = [line[1][0].upper().replace(' ', '') for line in result[0]]
        scores = [line[1][1] for line in result[0]]

        full_text_raw = ''.join(texts)
        full_text_paddle_formatted = self.format_license_paddle(full_text_raw)
        corrected_text = self.try_correct_plate_smart(full_text_paddle_formatted)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"PaddleOCR Output: '{full_text_raw}', Formatted: '{full_text_paddle_formatted}', Smart Corrected: '{corrected_text}', Score: {avg_score}")

        if self.license_complies_format_flexible(corrected_text):
            return corrected_text, avg_score
        return None, None 
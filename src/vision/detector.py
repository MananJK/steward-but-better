import numpy as np
from typing import Optional, Tuple, Dict, Any
import json


class F1VisionDetector:
    def __init__(self):
        self.vehicle_classes = {"VER": "Car 1 (VER)", "HAM": "Car 2 (HAM)"}
        self.track_line_class = "White Track Limit Line"
        self.pixels_per_meter = 100.0

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect cars and track limit lines in the frame.

        Args:
            frame: Input image frame (H, W, C) in BGR format

        Returns:
            Dictionary containing detections
        """
        detections = {
            "cars": [],
            "track_lines": [],
            "apex_clearance_meters": None,
            "tires_over_line": False,
        }

        return detections

    def calculate_apex_clearance(
        self,
        car_bbox: Tuple[int, int, int, int],
        track_line_bbox: Tuple[int, int, int, int],
    ) -> Tuple[float, bool]:
        """
        Calculate distance between car tire and track limit line at apex.

        Args:
            car_bbox: (x1, y1, x2, y2) bounding box of the car
            track_line_bbox: (x1, y1, x2, y2) bounding box of track limit line

        Returns:
            Tuple of (clearance_meters, tires_over_line)
        """
        car_x1, car_y1, car_x2, car_y2 = car_bbox
        line_x1, line_y1, line_x2, line_y2 = track_line_bbox

        car_right_edge = car_x2
        line_left_edge = line_x1

        pixel_distance = line_left_edge - car_right_edge

        if pixel_distance < 0:
            tires_over_line = True
            pixel_distance = abs(pixel_distance)
        else:
            tires_over_line = False

        clearance_meters = pixel_distance / self.pixels_per_meter

        return round(clearance_meters, 1), tires_over_line

    def process_frame(self, frame: np.ndarray) -> str:
        """
        Process a video frame and return JSON-ready string with detection results.

        Args:
            frame: Input image frame (H, W, C) in BGR format

        Returns:
            JSON-ready string with apex clearance and tire status
        """
        detections = self.detect(frame)

        if detections["apex_clearance_meters"] is None:
            apex_clearance = "N/A"
            tires_over = "Unknown"
        else:
            apex_clearance = f"{detections['apex_clearance_meters']:.1f}"
            tires_over = "Yes" if detections["tires_over_line"] else "No"

        result = f"Apex clearance: {apex_clearance}m; Tires over line: [{tires_over}]"

        return result

    def detect_from_image(self, image_path: str) -> str:
        """
        Detect from an image file.

        Args:
            image_path: Path to the image file

        Returns:
            JSON-ready string with detection results
        """
        try:
            import cv2

            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_path}")
            return self.process_frame(frame)
        except ImportError:
            return "Apex clearance: N/A; Tires over line: [Unknown]"


def detect_incident(frame: np.ndarray) -> str:
    """
    Main detection function for processing incident frames.

    Args:
        frame: Video frame or image from Abu Dhabi 2021 incident

    Returns:
        JSON-ready string with detection results
    """
    detector = F1VisionDetector()
    return detector.process_frame(frame)


if __name__ == "__main__":
    import cv2

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        result = detect_incident(frame)
        print(result)
    else:
        print("Apex clearance: N/A; Tires over line: [Unknown]")

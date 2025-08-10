import os
from typing import Any, Dict, List, Union

import cv2
import numpy as np


def adjust_exposure(img: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    """
    Adjust the exposure of an image.
    
    :param img: Input image
    :param alpha: Contrast control (1.0-3.0). Higher values increase exposure.
    :param beta: Brightness control (0-100). Higher values add brightness.
    :return: Exposure adjusted image
    """
    # Apply exposure adjustment using the formula: new_img = img * alpha + beta
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_img

def sharpen_image(img: np.ndarray) -> np.ndarray:
    """
    Apply a sharpening filter to an image.
    
    :param img: Input image
    :return: Sharpened image
    """
    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    # Apply the sharpening filter
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened


def plot_yolo_res(frame: np.ndarray, yolo_results: Dict[str, Any], 
                  conf_threshold: float = 0.5) -> np.ndarray:
    """
    Draw YOLO detection results on an image frame.
    
    :param frame: Input image frame (OpenCV format - BGR)
    :param yolo_results: YOLO detection results dictionary from gRPC client
    :param conf_threshold: Confidence threshold for displaying detections
    :return: Image frame with bounding boxes and labels drawn
    """
    # Create a copy to avoid modifying the original frame
    result_frame = frame.copy()
    
    # Check if results contain detections
    if 'detections' not in yolo_results or not yolo_results['detections']:
        return result_frame
    
    h, w = frame.shape[:2]
    
    # Define colors for different classes (BGR format for OpenCV)
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green  
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    for detection in yolo_results['detections']:
        # Extract detection information
        confidence = float(detection.get('confidence', 0.0))
        if confidence < conf_threshold:
            continue
            
        class_name = detection.get('class', 'unknown')
        
        # Get bounding box coordinates (assuming normalized coordinates 0-1)
        x_center = float(detection.get('x', 0.0))
        y_center = float(detection.get('y', 0.0))
        box_width = float(detection.get('width', 0.0))
        box_height = float(detection.get('height', 0.0))
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int((x_center - box_width / 2) * w)
        y1 = int((y_center - box_height / 2) * h)
        x2 = int((x_center + box_width / 2) * w)
        y2 = int((y_center + box_height / 2) * h)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # Choose color based on class hash
        color = colors[hash(class_name) % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        cv2.rectangle(result_frame, 
                     (x1, label_y - text_height - baseline), 
                     (x1 + text_width, label_y + baseline), 
                     color, -1)
        
        # Draw label text
        cv2.putText(result_frame, label, (x1, label_y - baseline), 
                   font, font_scale, (255, 255, 255), thickness)
    
    return result_frame

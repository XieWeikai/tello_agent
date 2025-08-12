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


def plot_yolo_res(frame: np.ndarray, yolo_results: List[Dict[str, Any]], 
                  conf_threshold: float = 0.5) -> np.ndarray:
    """Draw YOLO detection results on an image frame.
    
    Args:
        frame: Input image frame in OpenCV format (BGR color space).
        yolo_results: List of detection dictionaries containing name, confidence, and box coordinates.
            Each detection should have the format:
            {
                'name': str,
                'confidence': float,
                'box': {'x1': float, 'y1': float, 'x2': float, 'y2': float}
            }
        conf_threshold: Confidence threshold for displaying detections. Only detections
            with confidence >= conf_threshold will be drawn.
    
    Returns:
        Image frame with bounding boxes and labels drawn.
    """
    # Create a copy to avoid modifying the original frame
    result_frame = frame.copy()
    
    # Check if results contain detections
    if not yolo_results or len(yolo_results) == 0:
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
    
    for detection in yolo_results:
        # Extract detection information
        confidence = float(detection.get('confidence', 0.0))
        if confidence < conf_threshold:
            continue
            
        class_name = detection.get('name', 'unknown')
        box = detection.get('box', {})
        
        # Get bounding box coordinates (normalized coordinates 0-1)
        x1_norm = float(box.get('x1', 0.0))
        y1_norm = float(box.get('y1', 0.0))
        x2_norm = float(box.get('x2', 0.0))
        y2_norm = float(box.get('y2', 0.0))
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(x1_norm * w)
        y1 = int(y1_norm * h)
        x2 = int(x2_norm * w)
        y2 = int(y2_norm * h)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        # Calculate adaptive scaling based on image size
        # Use geometric mean of width and height for scaling
        image_scale = np.sqrt(w * h) / 1000.0  # Normalize to ~1000px reference
        
        # Adaptive box thickness (minimum 1, maximum 8)
        box_thickness = max(1, min(8, int(2 * image_scale)))
        
        # Adaptive font scale (minimum 0.3, maximum 1.5)
        font_scale = max(0.3, min(1.5, 0.5 * image_scale))
        
        # Adaptive text thickness (minimum 1, maximum 3)
        text_thickness = max(1, min(3, int(1 + image_scale * 0.5)))
        
        # Adaptive margin for text positioning
        text_margin = max(5, int(10 * image_scale))
        
        # Choose color based on class hash
        color = colors[hash(class_name) % len(colors)]
        
        # Draw bounding box with adaptive thickness
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, box_thickness)
        
        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate text size and position with adaptive font
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness)
        
        # Calculate label position with adaptive margin
        label_y = y1 - text_margin if y1 - text_margin > text_height else y1 + text_height + text_margin
        
        # Draw label background with adaptive size
        background_padding = max(2, int(3 * image_scale))
        cv2.rectangle(result_frame, 
                     (x1, label_y - text_height - baseline - background_padding), 
                     (x1 + text_width + background_padding * 2, label_y + baseline + background_padding), 
                     color, -1)
        
        # Draw label text with adaptive font
        cv2.putText(result_frame, label, 
                   (x1 + background_padding, label_y - baseline), 
                   font, font_scale, (255, 255, 255), text_thickness)
    
    return result_frame

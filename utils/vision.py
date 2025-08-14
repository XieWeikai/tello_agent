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


def format_yolo_res(yolo_results: List[Dict[str, Any]], 
                   conf_threshold: float = 0.5, 
                   sort_by_confidence: bool = True) -> str:
    """Format YOLO detection results into a human-readable string.
    
    Args:
        yolo_results: List of detection dictionaries containing name, confidence, and box coordinates.
            Each detection should have the format:
            {
                'name': str,
                'confidence': float,
                'box': {'x1': float, 'y1': float, 'x2': float, 'y2': float}
            }
        conf_threshold: Confidence threshold for including detections in the output.
        sort_by_confidence: Whether to sort detections by confidence in descending order.
    
    Returns:
        Formatted string containing detection information.
    """
    if not yolo_results or len(yolo_results) == 0:
        return "No objects detected."
    
    # Filter detections by confidence threshold
    filtered_detections = [
        detection for detection in yolo_results 
        if detection.get('confidence', 0.0) >= conf_threshold
    ]
    
    if not filtered_detections:
        return f"No objects detected above confidence threshold {conf_threshold:.2f}."
    
    # Sort by confidence if requested
    if sort_by_confidence:
        filtered_detections.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
    
    # Count objects by class
    class_counts = {}
    for detection in filtered_detections:
        class_name = detection.get('name', 'unknown').split('_')[0]  # Remove ID suffix
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Build formatted string
    result_lines = []
    
    # Summary line
    total_objects = len(filtered_detections)
    unique_classes = len(class_counts)
    result_lines.append(f"Detection Summary: {total_objects} objects detected ({unique_classes} unique classes)")
    result_lines.append("=" * 60)
    
    # Class summary
    result_lines.append("Object Count by Class:")
    for class_name, count in sorted(class_counts.items()):
        result_lines.append(f"  • {class_name}: {count}")
    result_lines.append("")
    
    # Detailed detection list
    result_lines.append("Detailed Detection Results:")
    result_lines.append("-" * 40)
    
    for i, detection in enumerate(filtered_detections, 1):
        name = detection.get('name', 'unknown')
        confidence = detection.get('confidence', 0.0)
        box = detection.get('box', {})
        
        # Extract box coordinates
        x1 = box.get('x1', 0.0)
        y1 = box.get('y1', 0.0)
        x2 = box.get('x2', 0.0)
        y2 = box.get('y2', 0.0)
        
        # Calculate box center and size
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Format detection info
        result_lines.append(f"{i:2d}. {name}")
        result_lines.append(f"    Confidence: {confidence:.1%}")
        result_lines.append(f"    Position: Center({center_x:.3f}, {center_y:.3f})")
        result_lines.append(f"    Size: {width:.3f} × {height:.3f} (Area: {area:.3f})")
        result_lines.append(f"    Bounding Box: ({x1:.3f}, {y1:.3f}) → ({x2:.3f}, {y2:.3f})")
        
        # Add position description
        pos_desc = []
        if center_x < 0.33:
            pos_desc.append("left")
        elif center_x > 0.67:
            pos_desc.append("right")
        else:
            pos_desc.append("center")
            
        if center_y < 0.33:
            pos_desc.append("top")
        elif center_y > 0.67:
            pos_desc.append("bottom")
        else:
            pos_desc.append("middle")
        
        result_lines.append(f"    Location: {' '.join(pos_desc)} of image")
        result_lines.append("")
    
    # Add statistics
    if len(filtered_detections) > 1:
        confidences = [d.get('confidence', 0.0) for d in filtered_detections]
        avg_conf = sum(confidences) / len(confidences)
        max_conf = max(confidences)
        min_conf = min(confidences)
        
        result_lines.append("Statistics:")
        result_lines.append(f"  Average Confidence: {avg_conf:.1%}")
        result_lines.append(f"  Confidence Range: {min_conf:.1%} - {max_conf:.1%}")
    
    return "\n".join(result_lines)



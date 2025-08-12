import cv2
import numpy as np
from djitellopy import Tello
from utils import vision


class Drone:
    def __init__(self):
        self.drone = Tello()
        self.drone.connect()
        
        self.drone.streamoff()
        self.drone.streamon()
        self.frame_reader = self.drone.get_frame_read()

    def get_frame(self, sharpen: bool=True) -> np.ndarray:
        """Get the current frame from the drone.

        Args:
            sharpen (bool, optional): Whether to apply sharpening and exposure adjustment. Defaults to True.

        Returns:
            np.ndarray: The processed frame as a NumPy array (RGB format).
        """
        fr = self.frame_reader.frame
        if not sharpen:
            return fr
        
        fr_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        fr_sharpened = vision.sharpen_image(vision.adjust_exposure(fr_bgr, alpha=1.3, beta=-30))
        fr_sharpened_rgb = cv2.cvtColor(fr_sharpened, cv2.COLOR_BGR2RGB)
        return fr_sharpened_rgb


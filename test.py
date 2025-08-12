import cv2
import dotenv
import numpy as np
from PIL import Image
from tello import Drone
from utils.vision import *

dotenv.load_dotenv()

from yolo.yolo_grpc_client import YoloGRPCClient

yolo_client = YoloGRPCClient()


def get_frame(detect: bool = True) -> np.ndarray:
    if getattr(get_frame, 'drone', None) is None:
        get_frame.drone = Drone()
    # RGB format
    fr = get_frame.drone.get_frame()

    pil_image = Image.fromarray(fr)
    
    if detect:
        # use yolo_client to detect objects
        yolo_results = yolo_client.detect_local(pil_image, conf=0.3)
        # draw detection results on the frame
        result_frame = plot_yolo_res(fr, yolo_results["result"], conf_threshold=0.3)
    else:
        result_frame = fr

    return result_frame

def tello_live_feed(detect: bool = True):
    while True:
        frame = get_frame(detect=detect)
        cv2.imshow("Live Feed", frame)

        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # tello_live_feed(False)
    tello_live_feed()

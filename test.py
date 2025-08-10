import cv2
import dotenv
import numpy as np
from PIL import Image
from tello import Drone
from utils.vision import *

dotenv.load_dotenv()

from yolo.yolo_grpc_client import YoloGRPCClient

drone = Drone()
yolo_client = YoloGRPCClient()


def get_frame():
    # 获取无人机原始帧 (numpy array, RGB格式)
    fr = drone.get_frame()
    
    # 图像增强处理需要先转换为BGR (OpenCV格式)
    fr_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
    sharpen_fr = sharpen_image(adjust_exposure(fr_bgr, alpha=1.3, beta=-30))
    
    # 将处理后的BGR图像转换为RGB，然后转为PIL Image用于YOLO检测
    sharpen_fr_rgb = cv2.cvtColor(sharpen_fr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(sharpen_fr_rgb)
    
    # YOLO 检测 (输入 PIL Image)
    yolo_results = yolo_client.detect_local(pil_image, conf=0.2)
    
    # 在BGR格式的numpy array上绘制检测框
    result_frame = plot_yolo_res(sharpen_fr, yolo_results, conf_threshold=0.5)
    
    # 返回 BGR 格式用于 cv2.imshow 显示
    return result_frame

while True:
    frame = get_frame()
    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

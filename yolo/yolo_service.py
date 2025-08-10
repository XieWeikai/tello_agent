import gc
import json
import multiprocessing
import os
import sys
from concurrent import futures
from io import BytesIO

import dotenv
import grpc
import torch
from PIL import Image

dotenv.load_dotenv()

from ultralytics import YOLO

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # tello_agent

ROOT_PATH = os.environ.get("ROOT_PATH", PARENT_DIR)
SERVICE_PORT = [port.strip() for port in os.environ.get("YOLO_SERVICE_PORT", "50050,50052").split(",")]

YOLO_PATH = os.path.join(ROOT_PATH, "yolo") # tello_agent/yolo
if YOLO_PATH not in sys.path:
    sys.path.append(YOLO_PATH)

if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

MODEL_PATH = os.path.join(ROOT_PATH, "models")
MODEL_TYPE = "yolov8x.pt"

from yolo import yolo_pb2, yolo_pb2_grpc


def load_model():
    model = YOLO(MODEL_PATH + MODEL_TYPE)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model.to(device)
    print(f"GPU memory usage: {torch.cuda.memory_allocated()}")
    return model

def release_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

"""
    gRPC service class.
"""
class YoloService(yolo_pb2_grpc.YoloServiceServicer):
    def __init__(self, port):
        self.stream_mode = False
        self.model = load_model()
        self.port = port

    def reload_model(self):
        if self.model is not None:
            release_model(self.model)
        self.model = load_model()

    @staticmethod
    def bytes_to_image(image_bytes):
        return Image.open(BytesIO(image_bytes))
    
    @staticmethod
    def format_result(yolo_result):
        if yolo_result.probs is not None:
            print('Warning: Classify task do not support `tojson` yet.')
            return
        formatted_result = []
        data = yolo_result.boxes.data.cpu().tolist()
        h, w = yolo_result.orig_shape
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            box = {'x1': round(row[0] / w, 2), 'y1': round(row[1] / h, 2), 'x2': round(row[2] / w, 2), 'y2': round(row[3] / h, 2)}
            conf = row[-2]
            class_id = int(row[-1])

            name = yolo_result.names[class_id]
            if yolo_result.boxes.is_track:
                # result['track_id'] = int(row[-3])  # track ID
                name = f'{name}_{int(row[-3])}'
            result = {'name': name, 'confidence': round(conf, 2), 'box': box}
            
            if yolo_result.masks:
                x, y = yolo_result.masks.xy[i][:, 0], yolo_result.masks.xy[i][:, 1]  # numpy array
                result['segments'] = {'x': (x / w).tolist(), 'y': (y / h).tolist()}
            if yolo_result.keypoints is not None:
                x, y, visible = yolo_result.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result['keypoints'] = {'x': (x / w).tolist(), 'y': (y / h).tolist(), 'visible': visible.tolist()}
            formatted_result.append(result)
        return formatted_result
    
    def process_image(self, image, id=None, conf=0.3):
        if self.stream_mode:
            yolo_result = self.model.track(image, verbose=False, conf=conf, tracker="bytetrack.yaml")[0]
        else:
            yolo_result = self.model(image, verbose=False, conf=conf)[0]
        result = {
            "image_id": id,
            "result": YoloService.format_result(yolo_result),
        }
        return json.dumps(result)

    def DetectStream(self, request, context):
        print(f"Received DetectStream request from {context.peer()} on port {self.port}, image_id: {request.image_id}")
        if not self.stream_mode:
            self.stream_mode = True
            self.reload_model()
        
        image = YoloService.bytes_to_image(request.image_data)
        return yolo_pb2.DetectResponse(json_data=self.process_image(image, request.image_id, request.conf))
    
    def Detect(self, request, context):
        print(f"Received Detect request from {context.peer()} on port {self.port}, image_id: {request.image_id}")
        if self.stream_mode:
            self.stream_mode = False
            self.reload_model()

        image = YoloService.bytes_to_image(request.image_data)
        return yolo_pb2.DetectResponse(json_data=self.process_image(image, request.image_id, request.conf))

def serve(port):
    print(f"Starting YoloService at port {port}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    yolo_pb2_grpc.add_YoloServiceServicer_to_server(YoloService(port), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    # Create a pool of processes
    process_count = len(SERVICE_PORT)
    processes = []

    for i in range(process_count):
        process = multiprocessing.Process(target=serve, args=(SERVICE_PORT[i],))
        process.start()
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.join()

# this is the expected response format for the Detect and DetectStream methods
# {
#   "image_id": 123,  // 来自请求的 image_id（可选）
#   "result": [
#     {
#       "name": "person_1",       // 类别名称（带跟踪ID时格式为"类别_ID"）
#       "confidence": 0.95,       // 置信度（保留2位小数）
#       "box": {                  // 归一化后的边界框坐标（相对图像宽高）
#         "x1": 0.1, "y1": 0.2,  // 左上角坐标
#         "x2": 0.3, "y2": 0.4    // 右下角坐标
#       },
#       "segments": {             // 可选：分割掩膜坐标（如果启用分割）
#         "x": [0.1, 0.2, ...],   // 归一化的x坐标列表
#         "y": [0.3, 0.4, ...]    // 归一化的y坐标列表
#       },
#       "keypoints": {            // 可选：关键点坐标（如果启用关键点检测）
#         "x": [0.1, 0.2, ...],  // 归一化的x坐标列表
#         "y": [0.3, 0.4, ...],  // 归一化的y坐标列表
#         "visible": [1, 0, ...]  // 关键点可见性（1可见，0不可见）
#       }
#     },
#     // 其他检测对象...
#   ]
# }

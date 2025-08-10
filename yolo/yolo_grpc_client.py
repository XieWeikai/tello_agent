import asyncio
import json
import os
import sys
from io import BytesIO
from typing import Any, Dict, Optional

import grpc
from PIL import Image

DEFAULT_YOLO_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # tello_agent
ROOT_PATH = os.environ.get("ROOT_PATH", PARENT_DIR)
YOLO_PATH = os.path.join(ROOT_PATH, "yolo") # tello_agent/yolo
print(f"YOLO_PATH: {YOLO_PATH}")
if YOLO_PATH not in sys.path:
    sys.path.append(YOLO_PATH)

from yolo import yolo_pb2, yolo_pb2_grpc

VISION_SERVICE_IP = os.environ.get("VISION_SERVICE_IP", "localhost")
YOLO_SERVICE_PORT = os.environ.get("YOLO_SERVICE_PORT", "50050").split(",")[0]

'''
Access the YOLO service through gRPC.
Simplified version without shared frame dependencies.
'''
class YoloGRPCClient():
    def __init__(self):
        # 同步通道 - 用于本地服务
        self.sync_channel = grpc.insecure_channel(f'{VISION_SERVICE_IP}:{YOLO_SERVICE_PORT}')
        self.stub = yolo_pb2_grpc.YoloServiceStub(self.sync_channel)
        
        # 异步通道 - 用于远程服务，延迟初始化
        self.async_channel = None
        self.stub_async = None
        
        self.image_size = (640, 352)
        self.frame_id = 0

    def _ensure_async_channel(self):
        """确保异步通道已初始化"""
        if self.async_channel is None:
            self.async_channel = grpc.aio.insecure_channel(f'{VISION_SERVICE_IP}:{YOLO_SERVICE_PORT}')
            self.stub_async = yolo_pb2_grpc.YoloServiceStub(self.async_channel)

    def is_local_service(self) -> bool:
        return VISION_SERVICE_IP == 'localhost'

    @staticmethod
    def image_to_bytes(image: Image.Image) -> bytes:
        """Convert PIL Image to bytes with WEBP compression"""
        imgByteArr = BytesIO()
        image.save(imgByteArr, format='WEBP')
        return imgByteArr.getvalue()
    
    def detect_local(self, image: Image.Image, conf: float = 0.2) -> Dict[str, Any]:
        """Synchronous local detection"""
        image_bytes = self.image_to_bytes(image.resize(self.image_size))
        
        detect_request = yolo_pb2.DetectRequest(image_data=image_bytes, conf=conf)
        response = self.stub.DetectStream(detect_request)
        
        return json.loads(response.json_data)

    async def detect_remote(self, image: Image.Image, conf: float = 0.1) -> Dict[str, Any]:
        """Asynchronous remote detection"""
        self._ensure_async_channel()

        # Use original image size for remote detection
        image_bytes = self.image_to_bytes(image)
        
        detect_request = yolo_pb2.DetectRequest(
            image_id=self.frame_id, 
            image_data=image_bytes, 
            conf=conf
        )
        self.frame_id += 1
        
        response = await self.stub_async.Detect(detect_request)
        return json.loads(response.json_data)

    async def detect(self, image: Image.Image, conf: float = 0.1) -> Dict[str, Any]:
        """
        Main detection method that automatically chooses local or remote detection
        
        Args:
            image: PIL Image to detect objects in
            conf: Confidence threshold for detection
            
        Returns:
            Dictionary containing detection results in JSON format
        """
        if self.is_local_service():
            return self.detect_local(image, conf)
        else:
            return await self.detect_remote(image, conf)

    def detect_sync(self, image: Image.Image, conf: float = 0.2) -> Dict[str, Any]:
        """
        Synchronous detection method for easier use in non-async contexts
        
        Args:
            image: PIL Image to detect objects in
            conf: Confidence threshold for detection
            
        Returns:
            Dictionary containing detection results in JSON format
        """
        if self.is_local_service():
            return self.detect_local(image, conf)
        else:
            # Run async method in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.detect_remote(image, conf))
            finally:
                loop.close()
                
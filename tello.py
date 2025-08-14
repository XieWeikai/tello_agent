import inspect
import threading
import time
from functools import wraps

import cv2
import numpy as np
from djitellopy import Tello
from smolagents import tool

from utils import vision


def expose_methods_as_tools(*, include: list = None, exclude: list = None):
    """
    类装饰器：将指定方法暴露为smolagents工具
    
    参数:
        include: 要暴露为工具的方法名列表（如果为None，则包含所有公共方法）
        exclude: 要排除的方法名列表
    
    示例:
        @expose_methods_as_tools(include=['method1', 'method2'])
        @expose_methods_as_tools(exclude=['internal_method'])
    """
    def decorator(cls):
        class ToolWrapper:
            def __init__(self, instance):
                self.instance = instance
                self.tools = []
                
                # 确定要处理的方法
                methods_to_process = []
                for attr_name in dir(instance):
                    attr = getattr(instance, attr_name)
                    if callable(attr) and not attr_name.startswith("_"):
                        # 应用包含/排除规则
                        if include and attr_name not in include:
                            continue
                        if exclude and attr_name in exclude:
                            continue
                        methods_to_process.append(attr_name)
                
                # 为每个方法创建工具
                for method_name in methods_to_process:
                    method = getattr(instance, method_name)
                    
                    # 创建工具函数（保留原始签名和文档）
                    @wraps(method)
                    def tool_func(*args, __method=method, **kwargs):
                        return __method(*args, **kwargs)
                    
                    # 应用@tool装饰器并保留所有元数据
                    tool_func = tool(tool_func)
                    
                    # 确保工具名称唯一（避免冲突）
                    tool_func.__name__ = f"{instance.__class__.__name__}_{method_name}"
                    self.tools.append(tool_func)
        
        # 原始类的初始化包装
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._tool_wrapper = ToolWrapper(self)
        
        cls.__init__ = new_init
        
        # 添加获取工具的方法
        cls.get_tools = lambda self: self._tool_wrapper.tools
        
        return cls
    
    return decorator


@expose_methods_as_tools(include=[
    'move_forward',
    'take_off',
    'move_up',
    'move_down',
    'move_left',
    'move_right',
    'turn_clockwise',
    'turn_counter_clockwise',
    'land'
], exclude=['get_frame'])
class Drone:
    def __init__(self):
        self.drone = Tello()
        self.drone_lock = threading.Lock()  # 添加无人机操作锁
        
        with self.drone_lock:
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
        with self.drone_lock:
            fr = self.frame_reader.frame
        
        if not sharpen:
            return fr
        
        fr_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        fr_sharpened = vision.sharpen_image(vision.adjust_exposure(fr_bgr, alpha=1.3, beta=-30))
        fr_sharpened_rgb = cv2.cvtColor(fr_sharpened, cv2.COLOR_BGR2RGB)
        return fr_sharpened_rgb
    
    def move_forward(self, distance: int) -> None:
        """
        Move the drone forward by a specified distance in centimeters.
        
        Args:
            distance (int): The distance to move forward in centimeters.
        """
        with self.drone_lock:
            self.drone.move_forward(distance)
        time.sleep(0.5)
    
    def land(self) -> None:
        """
        Land the drone.
        """
        with self.drone_lock:
            self.drone.land()
    
    def take_off(self) -> None:
        """
        Take off the drone.
        """
        with self.drone_lock:
            self.drone.takeoff()
        time.sleep(0.5)
    
    def move_up(self, distance: int) -> None:
        """
        This is a tool that moves the drone up by a specified distance in centimeters.
        
        Args:
            distance (int): The distance to move up in centimeters.
        """
        with self.drone_lock:
            self.drone.move_up(distance)
        time.sleep(0.5)

    def move_down(self, distance: int) -> None:
        """
        Move the drone down by a specified distance in centimeters.
        
        Args:
            distance (int): The distance to move down in centimeters.
        """
        with self.drone_lock:
            self.drone.move_down(distance)
        time.sleep(0.5)

    def move_left(self, distance: int) -> None:
        """
        Move the drone left by a specified distance in centimeters.

        Args:
            distance (int): The distance to move left in centimeters.
        """
        with self.drone_lock:
            self.drone.move_left(distance)
        time.sleep(0.5)

    def move_right(self, distance: int) -> None:
        """
        Move the drone right by a specified distance in centimeters.

        Args:
            distance (int): The distance to move right in centimeters.
        """
        with self.drone_lock:
            self.drone.move_right(distance)
        time.sleep(0.5)

    def turn_clockwise(self, degrees: int) -> None:
        """
        Turn the drone clockwise by a specified number of degrees.

        Args:
            degrees (int): The number of degrees to turn clockwise.
        """
        with self.drone_lock:
            self.drone.rotate_clockwise(degrees)
        time.sleep(0.5)

    def turn_counter_clockwise(self, degrees: int) -> None:
        """
        Turn the drone counter-clockwise by a specified number of degrees.

        Args:
            degrees (int): The number of degrees to turn counter-clockwise.
        """
        with self.drone_lock:
            self.drone.rotate_counter_clockwise(degrees)
        time.sleep(0.5)

    def detect_objects(self, conf: float=0.5) -> str:
        """
        Detect objects in the current frame using YOLO and return a description of the detected objects.

        Args:
            conf (float): Confidence threshold for object detection.

        Returns:
            str: A message indicating the detected objects.
        """
        frame = self.get_frame()
        yolo_results = vision.yolo_detect(frame, conf=conf)
        
        if not yolo_results:
            return "No objects detected."
        
        return f"Objects detected: {', '.join([res['name'] for res in yolo_results])}"
    
    def get_battery(self) -> int:
        """
        Get the current battery percentage.
        
        Returns:
            int: Battery percentage (0-100)
        """
        with self.drone_lock:
            return self.drone.get_battery()
    
    def set_speed(self, speed: int) -> None:
        """
        Set the drone's speed.
        
        Args:
            speed (int): Speed in cm/s (10-100)
        """
        with self.drone_lock:
            self.drone.set_speed(speed)
        

@expose_methods_as_tools()
class FakeDrone:
    def move_up(self, distance: int) -> None:
        print(f"FakeDrone: Moving up {distance} cm")

    def move_down(self, distance: int) -> None:
        print(f"FakeDrone: Moving down {distance} cm")

    def move_left(self, distance: int) -> None:
        print(f"FakeDrone: Moving left {distance} cm")

    def move_right(self, distance: int) -> None:
        print(f"FakeDrone: Moving right {distance} cm")

    def turn_clockwise(self, degrees: int) -> None:
        print(f"FakeDrone: Turning clockwise {degrees} degrees")

    def turn_counter_clockwise(self, degrees: int) -> None:
        print(f"FakeDrone: Turning counter-clockwise {degrees} degrees")
        

if __name__ == "__main__":
    
    @expose_methods_as_tools(include=['fly_up', 'detect_objects'])
    class TestTools:
        def fly_up(self, distance: int) -> None:
            """
            This is a test tool that simulates flying up by a specified distance.
            Args:
                distance (int): The distance to fly up in centimeters.
            """
            print(f"Flying up {distance} cm")

        def detect_objects(self) -> str:
            """
            This is a test tool that simulates object detection.
            Returns:
                str: A message indicating objects detected.
            """
            return "Objects detected: [tree, car, person]"
        
    
    drone = Drone()
    tools = drone.get_tools()
    # tools = TestTools().get_tools()
    
    from smolagents import CodeAgent, InferenceClientModel
    agent = CodeAgent(tools=tools, model=InferenceClientModel())
    # agent = CodeAgent(tools=tools, model=InferenceClientModel("Qwen/Qwen2.5-VL-32B-Instruct"))
    print(agent.memory.system_prompt.system_prompt)
    agent.run("起飞，往前飞25cm然后降落")
        
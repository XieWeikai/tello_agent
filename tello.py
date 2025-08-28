import threading
import time
from functools import wraps

import cv2
import numpy as np
from djitellopy import Tello
from PIL import Image
from smolagents import tool

from utils import vision
from yolo.yolo_grpc_client import YoloGRPCClient


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


@expose_methods_as_tools(
    include=[
        "move_forward",
        "take_off",
        "move_up",
        "move_down",
        "move_left",
        "move_right",
        "turn_clockwise",
        "turn_counter_clockwise",
        "land",
    ],
    exclude=["get_frame"],
)
class Drone:
    def __init__(self):
        self.drone = Tello()
        self.start_frame_reader = False
        self.fr_state = None
        self.fps = None
        self.yolo_client = YoloGRPCClient()
        self.drone_lock = threading.Lock()
        self.is_take_off = False

        with self.drone_lock:
            self.drone.connect()

            self.drone.streamoff()
            self.drone.streamon()
            self.frame_reader = self.drone.get_frame_read()

    def start_frame_thread(self, fps=30, detect=False):
        """start a thread to continuously update the drone's video frame"""
        self.stop_updater = threading.Event()
        def frame_updater():
            while not self.stop_updater.is_set():
                with self.drone_lock:
                    fr = self.frame_reader.frame
                
                fr_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                fr_sharpened = vision.sharpen_image(
                    vision.adjust_exposure(fr_bgr, alpha=1.3, beta=-30)
                )
                fr_sharpened_rgb = cv2.cvtColor(fr_sharpened, cv2.COLOR_BGR2RGB)
                detections = None
                if detect:
                    pil_image = Image.fromarray(fr_sharpened_rgb)
                    detections = self.yolo_client.detect_local(pil_image, conf=0.3)["result"]
                self.fr_state = (fr_sharpened_rgb, detections)
                
                time.sleep(1.0 / fps)

        self.frame_update_thread = threading.Thread(target=frame_updater)
        self.frame_update_thread.start()
        self.start_frame_reader = True
        self.fps = fps

    def stop_frame_thread(self):
        """Stop the frame update thread."""
        self.stop_updater.set()
        self.frame_update_thread.join()
        self.start_frame_reader = False

    def get_frame(self) -> np.ndarray:
        """Get the current frame from the drone.

        Args:
            sharpen (bool, optional): Whether to apply sharpening and exposure adjustment. Defaults to True.

        Returns:
            np.ndarray: The processed frame as a NumPy array (RGB format).
        """
        assert self.start_frame_reader, "Frame reader thread is not started. Call start_frame_thread() first."
        while self.fr_state is None:
            time.sleep(0.01)
        fr, _ = self.fr_state

        return fr

    def live_feed(self, fps: int = 30, plot_detections: bool = False):
        """
        Display live video feed from the drone with optional object detection overlay.
        Press 'q' to exit and land the drone if it's flying.
        
        Args:
            fps (int): Frames per second for the video feed. Defaults to 30.
            plot_detections (bool): Whether to plot YOLO detection results on the frame. Defaults to True.
        """
        print("Starting live feed... Press 'q' to exit")
        
        try:
            while True:
                # Get current frame and detections from the frame state
                if self.fr_state is not None:
                    frame, detections = self.fr_state

                    if frame is not None:
                        # Convert RGB to BGR for OpenCV display
                        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # Plot detections if requested and available
                        if plot_detections:
                            assert detections is not None, "Detections should not be None when plotting is enabled. You should set detect=True in start_frame_thread()."
                            from utils.vision import plot_yolo_res
                            display_frame = plot_yolo_res(display_frame, detections, conf_threshold=0.3)
                        
                        # Get battery status for window title
                        battery = self.get_battery()
                        window_title = f"Tello Live Feed - Battery: {battery}%"
                        
                        # Display the frame
                        cv2.imshow(window_title, display_frame)
                
                # Check for 'q' key press to exit
                if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
                    print("Exiting live feed...")
                    # Land the drone if it's currently flying
                    if self.is_take_off:
                        print("Landing drone...")
                        self.land()
                    break
                    
        except Exception as e:
            print(f"Live feed error: {e}")
        finally:
            cv2.destroyAllWindows()
            print("Live feed terminated")

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
        self.is_take_off = False

    def take_off(self) -> None:
        """
        Take off the drone.
        """
        with self.drone_lock:
            self.drone.takeoff()
        time.sleep(0.5)
        self.is_take_off = True

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

    def detect_objects(self, conf: float = 0.5) -> str:
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
    
    def send_rc_control(self, left_right_velocity: int = 0, forward_backward_velocity: int = 0, 
                       up_down_velocity: int = 0, yaw_velocity: int = 0):
        """
        Send RC control command to drone.
        
        Args:
            left_right_velocity (int): left/right velocity (-100~100)
            forward_backward_velocity (int): forward/backward velocity (-100~100) 
            up_down_velocity (int): up/down velocity (-100~100)
            yaw_velocity (int): yaw velocity (-100~100)
        """
        with self.drone_lock:
            self.drone.send_rc_control(left_right_velocity, forward_backward_velocity, 
                                     up_down_velocity, yaw_velocity)
            
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


@expose_methods_as_tools(
    include=[
        "move_forward",
        "take_off",
        "move_up",
        "move_down",
        "move_left",
        "move_right",
        "turn_clockwise",
        "turn_counter_clockwise",
        "land",
    ],
    exclude=["get_frame"],
)
class MockDrone(Drone):
    def __init__(self):
        self.drone = Tello()
        self.drone_lock = threading.Lock()  # 添加无人机操作锁
        self.is_take_off = False
        self.start_frame_reader = False
        self.fr_state = None
        # with self.drone_lock:
        #     self.drone.connect()

        #     self.drone.streamoff()
        #     self.drone.streamon()
        #     self.frame_reader = self.drone.get_frame_read()

    def get_battery(self) -> int:
        return 100

    def connect(self):
        """Mock connect: does nothing."""
        pass

    def get_frame(self):
        """Mock get_frame: returns a blank image as a NumPy array."""
        # Create a black image (height, width, channels)
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def start_frame_thread(self, fps=30, detect=False):
        """Mock start_frame_thread: creates a simple frame state."""
        self.start_frame_reader = True
        # Create a simple black frame for mock
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.fr_state = (mock_frame, [] if detect else None)
    
    def land(self) -> None:
        """Mock land: does nothing."""
        self.is_take_off = False
        print("Mock drone landed")
    
    def take_off(self) -> None:
        """Mock takeoff: does nothing.""" 
        self.is_take_off = True
        print("Mock drone took off")
    
    def move_forward(self, distance: int) -> None:
        """Mock move_forward: does nothing."""
        print(f"Mock drone moved forward {distance}cm")
    
    def move_up(self, distance: int) -> None:
        """Mock move_up: does nothing."""
        print(f"Mock drone moved up {distance}cm")
    
    def move_down(self, distance: int) -> None:
        """Mock move_down: does nothing."""
        print(f"Mock drone moved down {distance}cm")
    
    def move_left(self, distance: int) -> None:
        """Mock move_left: does nothing."""
        print(f"Mock drone moved left {distance}cm")
    
    def move_right(self, distance: int) -> None:
        """Mock move_right: does nothing."""
        print(f"Mock drone moved right {distance}cm")
    
    def turn_clockwise(self, degrees: int) -> None:
        """Mock turn_clockwise: does nothing."""
        print(f"Mock drone turned clockwise {degrees} degrees")
    
    def turn_counter_clockwise(self, degrees: int) -> None:
        """Mock turn_counter_clockwise: does nothing."""
        print(f"Mock drone turned counter-clockwise {degrees} degrees")
    
    def send_rc_control(self, left_right_velocity: int = 0, forward_backward_velocity: int = 0,
                       up_down_velocity: int = 0, yaw_velocity: int = 0):
        """Mock send_rc_control: does nothing."""
        pass
    
    def set_speed(self, speed: int) -> None:
        """Mock set_speed: does nothing."""
        pass


if __name__ == "__main__":
    drone = Drone()
    time.sleep(0.5)
    bat = drone.get_battery()
    print(f"Battery: {bat}%")
    time.sleep(2.0)
    # drone.take_off()
    # time.sleep(0.5)
    # drone.turn_clockwise(90)
    # time.sleep(0.5)
    drone.land()
    time.sleep(0.5)

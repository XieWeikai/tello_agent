import threading
import time

import cv2
from PIL import Image

from tello import Drone
from utils.pid import PID
from utils.vision import plot_yolo_res
from yolo.yolo_grpc_client import YoloGRPCClient

yolo_client = YoloGRPCClient()

drone = Drone()

GAINS_YAW = dict(kp=150.0, ki=0.0, kd=0.0)   
GAINS_Z   = dict(kp=100.0, ki=0.02, kd=5.0)   # up/down
GAINS_FB  = dict(kp=100.0, ki=0.01, kd=15.0) # forward/backward

AREA_DESIRED = 0.4  # Desired area of the object in the frame (normalized, e.g., 0.4 means 40% of frame area)

YAW_DEADBAND = 0.003
Z_DEADBAND = 0.003
FB_DEADBAND = 0.003

FPS = 2

OBJ_NAME = "teddy"

frame_state = (None, None)  # (frame, detections)

is_take_off = False

def get_frame_and_detections():
    frame = drone.get_frame()  # Use our Drone class method
    pil_image = Image.fromarray(frame)

    detections = yolo_client.detect_local(pil_image, conf=0.3)["result"]

    return frame, detections

def start_frame_thread():
    def frame_thread():
        global frame_state
        while True:
            frame, detections = get_frame_and_detections()
            frame_state = (frame, detections)
            time.sleep(1.0 / FPS)

    t = threading.Thread(target=frame_thread, daemon=True)
    t.start()


def get_bbox(detections, class_name=OBJ_NAME):
    for det in detections:
        if class_name in det['name']:
            return det['box']["x1"], det['box']["y1"], det['box']["x2"], det['box']["y2"]  # [x1, y1, x2, y2]
    return None

def track_obj(fps=FPS, obj_name=OBJ_NAME):
    # global is_take_off
    # if not is_take_off:
    #     drone.take_off()
    #     is_take_off = True
    #     time.sleep(1)
    
    start_time = time.time()
    
    yaw_pid = PID(**GAINS_YAW, out_limit=100.0, i_limit=20.0, deadband=0.03)
    z_pid = PID(**GAINS_Z, out_limit=100.0, i_limit=20.0, deadband=0.03)
    fb_pid = PID(**GAINS_FB, out_limit=100.0, i_limit=20.0, deadband=0.03)
    
    not_seen_time = 0.0
    while True:
        cur_time = time.time()
        dt = (cur_time - start_time)
        start_time = cur_time
        
        frame, detections = frame_state
        if frame is None or detections is None:
            start_time = cur_time
            continue

        bbox = get_bbox(detections, class_name=obj_name)
        print(f"detections: {detections}")
        print(f"{obj_name} bbox: {bbox}")

        if bbox:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)

            # Compute errors for PID controllers
            err_yaw = cx - 0.5 if abs(cx - 0.5) > YAW_DEADBAND else 0.0  # Horizontal error (normalized)
            err_z = cy - 0.5 if abs(cy - 0.5) > Z_DEADBAND else 0.0  # Vertical error (normalized)
            err_fb = area - AREA_DESIRED if abs(area - AREA_DESIRED) > FB_DEADBAND else 0.0  # Forward/backward error (area-based)

            # Update PID controllers
            yaw_output = yaw_pid.step(err_yaw, dt=dt)
            z_output = z_pid.step(err_z, dt=dt)
            fb_output = fb_pid.step(err_fb, dt=dt)
            
            yaw_output = 0 if err_yaw == 0.0 else yaw_output
            z_output = 0 if err_z == 0.0 else z_output
            fb_output = 0 if err_fb == 0.0 else fb_output
            
            # just test yaw for now
            z_output = 0
            fb_output = 0

            # Send commands to drone
            if is_take_off:
                drone.send_rc_control(left_right_velocity=0,
                                      forward_backward_velocity=int(fb_output),
                                      up_down_velocity=int(z_output),
                                      yaw_velocity=int(yaw_output))
            not_seen_time = 0.0  # Reset not seen timer
        else:
            # No object detected, hover in place
            not_seen_time += dt
            if is_take_off:
                drone.send_rc_control(0, 0, 0, 0)
        
            if not_seen_time > 60.0:  # If not seen for 60 seconds, land
                print("Object lost for 60 seconds, stop tracking")
                break

        time.sleep(1.0 / fps)
        

def live_feed(fps=30, plot_detections=True):
    while True:
        frame, detections = frame_state
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        battery = drone.get_battery()
        # set cv2 window title to show battery
        cv2.setWindowTitle("Live Feed", f"Tello Live Feed - Battery: {battery}%")

        if plot_detections and frame is not None and detections is not None:
            frame = plot_yolo_res(frame, detections, conf_threshold=0.3)
            
        if frame is not None:
            cv2.imshow("Live Feed", frame)

        if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
            drone.land()
            is_take_off = False
            print("Landing and exiting live feed")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_frame_thread()
    threading.Thread(target=track_obj, daemon=True).start()
    live_feed()

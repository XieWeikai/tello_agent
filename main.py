import os
import threading
import time

import cv2
import dotenv
from PIL import Image
from smolagents import (ActionStep, CodeAgent, InferenceClientModel,
                        LiteLLMModel, ToolCallingAgent)

from tello import Drone

dotenv.load_dotenv()

BASE_URL = os.environ["BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))

AGENT_MODE = os.environ.get("AGENT_MODE", "code")
assert AGENT_MODE in ["code", "tool"], "AGENT_MODE must be 'code' or 'tool'"
AGENT_CLS = CodeAgent if AGENT_MODE == "code" else ToolCallingAgent


drone = Drone()

def keep_drone_alive():
    """定期发送keepalive命令防止自动降落"""
    while True:
        try:
            battery = drone.get_battery()
            print(f"Battery: {battery}%")
        except Exception as e:
            print(f"Keepalive error: {e}")
        time.sleep(10)

def tello_live_feed():
    """视频流显示线程"""
    print("Starting video feed...")
    try:
        while True:
            frame = drone.get_frame(sharpen=False)  # 获取原始帧用于显示
            # 转换为BGR格式供OpenCV显示
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Tello Live Feed", frame_bgr)

            # 按'q'键退出视频流
            if cv2.waitKey(30) & 0xFF == ord('q'):
                print("Video feed stopped by user")
                break
                
    except Exception as e:
        print(f"Video feed error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Video feed thread terminated")

# 启动keepalive线程
keepalive_thread = threading.Thread(target=keep_drone_alive)
keepalive_thread.daemon = True
keepalive_thread.start()

# 启动视频流线程
video_thread = threading.Thread(target=tello_live_feed)
video_thread.daemon = True
video_thread.start()

def update_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    """更新agent记忆中的截图"""
    latest_step = memory_step.step_number
    for previous_memory_step in agent.memory.steps:  
        # Remove previous screenshots from logs for lean processing
        if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= latest_step - 3:
            previous_memory_step.observations_images = None
    
    # 获取当前帧
    image = drone.get_frame(sharpen=True)  # 使用锐化后的帧用于AI分析
    pil_image = Image.fromarray(image)
    pil_image.show(title=f"Step {memory_step.step_number}")
    memory_step.observations_images = [pil_image.copy()]

if __name__ == "__main__":
    print("Initializing Tello Agent...")
    print("Video feed window will open automatically")
    print("Press 'q' in the video window to stop video feed")
    
    tools = drone.get_tools()
    
    model = LiteLLMModel(
        model_id=MODEL_NAME,
        api_base=BASE_URL,
        api_key=API_KEY,
        temperature=TEMPERATURE,  
    )

    agent = AGENT_CLS(
        tools=tools,
        model=model,
        step_callbacks=[update_screenshot])
    
    print("System prompt:")
    print(agent.memory.system_prompt.system_prompt)
    
    task_image = drone.get_frame()
    pil_image = Image.fromarray(task_image)
    pil_image.show(title="Task Image")
    
    try:
        print("\nStarting agent task...")
        agent.run("你背后有个白板，你看看上面有什么字，看不到的话你需要找到这个字", images=[pil_image])
    except Exception as e:
        print(f"Error occurred: {e}")
        agent.replay()
    finally:
        print("\nTask completed. Video feed and keepalive threads will continue running...")
        print("Press Ctrl+C to exit or 'q' in video window to stop video feed")
    

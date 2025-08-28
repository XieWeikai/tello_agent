import os
import threading
import time

import cv2
import dotenv
import yaml
from PIL import Image
from smolagents import (ActionStep, CodeAgent, InferenceClientModel,
                        LiteLLMModel, ToolCallingAgent)

from tello import Drone, MockDrone


class MyToolCallingAgent(ToolCallingAgent):
    @property
    def tools_and_managed_agents(self):
        """Returns a combined list of tools and managed agents."""
        return list(self.managed_agents.values())


dotenv.load_dotenv()

# BASE_URL = os.environ.get("BASE_URL", )
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "openrouter/anthropic/claude-sonnet-4")

TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))

AGENT_MODE = os.environ.get("AGENT_MODE", "code")

# USE_CUSTOM_PROMPT = os.environ.get("USE_CUSTOM_PROMPT", "false").lower() == "true"
USE_CUSTOM_PROMPT = True
PROMPT_PATH = os.environ.get("PROMPT_PATH", "code_agent.yaml")

AGENT_CLASS_MAP = {
    "code": CodeAgent,
    "tool": ToolCallingAgent,
    "prompt_tool": MyToolCallingAgent,
}
assert (
    AGENT_MODE in AGENT_CLASS_MAP
), "AGENT_MODE must be one of 'code', 'tool', or 'prompt_tool'"
AGENT_CLS = AGENT_CLASS_MAP[AGENT_MODE]

# USE_MOCK_DRONE = os.environ.get("USE_MOCK_DRONE", "false").lower() == "true"
USE_MOCK_DRONE = False
drone = MockDrone() if USE_MOCK_DRONE else Drone()


def keep_drone_alive():
    """定期发送keepalive命令防止自动降落"""
    while True:
        try:
            battery = drone.get_battery()
            drone.set_speed(10)  # 设置速度以保持连接
            print(f"Battery: {battery}%")
        except Exception as e:
            print(f"Keepalive error: {e}")
        time.sleep(1)

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
            if cv2.waitKey(30) & 0xFF == ord("q"):
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


def update_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    """更新agent记忆中的截图"""
    latest_step = memory_step.step_number
    for previous_memory_step in agent.memory.steps:
        # Remove previous screenshots from logs for lean processing
        if (
            isinstance(previous_memory_step, ActionStep)
            and previous_memory_step.step_number <= latest_step - 3
        ):
            previous_memory_step.observations_images = None

    # 获取当前帧
    image = drone.get_frame(sharpen=True)
    pil_image = Image.fromarray(image)
    # pil_image.show(title=f"Step {memory_step.step_number}")
    memory_step.observations_images = [pil_image.copy()]


def agent_main():
    # read from yaml path
    if USE_CUSTOM_PROMPT and PROMPT_PATH:
        if os.path.exists(PROMPT_PATH):
            with open(PROMPT_PATH, "r") as f:
                prompt_template = yaml.safe_load(f)
        else:
            print(f"Warning: {PROMPT_PATH} not found, using default prompt template")
            prompt_template = None
    else:
        prompt_template = None

    tools = drone.get_tools()

    model = LiteLLMModel(
        model_id=MODEL_NAME,
        # api_base=BASE_URL,
        api_key=API_KEY,
        # temperature=TEMPERATURE,
    )

    agent = AGENT_CLS(
        tools=tools,
        model=model,
        step_callbacks=[update_screenshot],
        planning_interval=8,
        prompt_templates=prompt_template,
    )

    print(f"{agent.memory.system_prompt.system_prompt}")

    task_image = drone.get_frame()
    pil_image = Image.fromarray(task_image)
    # pil_image.show(title="Task Image")

    try:
        print("\nStarting agent task...")
        # agent.run("Check the time on the alarm clock on the wall behind you", images=[pil_image])
        # agent.run("Find the electronic blackboard and calculate the result of the equation on the blackboard", images=[pil_image])
        agent.run("Move more in one direction until there are many tables. Find the electronic blackboard and calculate the result of the equation on the blackboard", images=[pil_image])
        # agent.run("Count how many chairs are on each side of the conference room", images=[pil_image])
    except Exception as e:
        print(f"Error occurred: {e}")
        agent.replay()
    finally:
        print(
            "\nTask completed. Video feed and keepalive threads will continue running..."
        )
        print("Press Ctrl+C to exit or 'q' in video window to stop video feed")


agent_main_thread = threading.Thread(target=agent_main)
agent_main_thread.daemon = True
agent_main_thread.start()

if __name__ == "__main__":
    tello_live_feed()

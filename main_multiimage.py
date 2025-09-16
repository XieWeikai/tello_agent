import os
import time
import dotenv
import yaml
import threading
import queue

from PIL import Image
from tello import MockDrone, Drone
from smolagents import ActionStep, CodeAgent, LiteLLMModel, ToolCallingAgent


class MyToolCallingAgent(ToolCallingAgent):
    @property
    def tools_and_managed_agents(self):
        """Returns a combined list of tools and managed agents."""
        return list(self.managed_agents.values())


dotenv.load_dotenv()
FPS = 15
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "openrouter/anthropic/claude-sonnet-4")
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))
AGENT_MODE = os.environ.get("AGENT_MODE", "code")
USE_CUSTOM_PROMPT = os.environ.get("USE_CUSTOM_PROMPT", "false").lower() == "true"
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
TASK_DESCRIPTION = os.environ.get("TASK_DESCRIPTION", "Find the electronic blackboard.")
USE_MOCK_DRONE = os.environ.get("USE_MOCK_DRONE", "false").lower() == "true"
drone = MockDrone() if USE_MOCK_DRONE else Drone()


stop_event = threading.Event()
frame_queue = queue.Queue(maxsize=10)


def keep_drone_alive():
    while not stop_event.is_set():
        try:
            battery = drone.get_battery()
            drone.set_speed(10)
            print(f"[KeepAlive] Battery: {battery}%")
        except Exception as e:
            print(f"Keepalive error: {e}")
        stop_event.wait(1)


def capture_frames():
    while not stop_event.is_set():
        frame = drone.get_frame()
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
        stop_event.wait(1)


def update_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    LAST_STEPS = 3
    IMAGE_PER_STEP = 3

    latest_step = memory_step.step_number
    for previous_memory_step in agent.memory.steps:
        # Remove previous screenshots from logs for lean processing
        if (
            isinstance(previous_memory_step, ActionStep)
            and previous_memory_step.step_number <= latest_step - LAST_STEPS
        ):
            previous_memory_step.observations_images = None

    if not frame_queue.empty():
        queue_list = list(frame_queue.queue)
        num_frames = min(IMAGE_PER_STEP, len(queue_list))
        latest_frames = queue_list[-num_frames:]
        pil_images = [Image.fromarray(frame) for frame in latest_frames]
        print(f"Got {len(pil_images)} frames.")
    memory_step.observations_images = pil_images


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
        api_key=API_KEY,
        temperature=TEMPERATURE,
    )
    agent = AGENT_CLS(
        tools=tools,
        model=model,
        step_callbacks=[update_screenshot],
        planning_interval=8,
        prompt_templates=prompt_template,
    )
    print(f"system_prompt = \n{agent.memory.system_prompt.system_prompt}")

    try:
        print("Start agent task.")
        frame = drone.get_frame()
        pil_image = Image.fromarray(frame)
        result = agent.run(TASK_DESCRIPTION, images=[pil_image])
    except Exception as e:
        print(f"Error occurred: {e}")
        agent.replay()
    finally:
        print("Task completed.")
        stop_event.set()


def main():
    keepalive_thread = threading.Thread(target=keep_drone_alive)
    capture_thread = threading.Thread(target=capture_frames)

    # drone.start_frame_thread(fps=FPS, detect=True)
    # drone.live_feed(FPS, plot_detections=True)
    keepalive_thread.start()
    capture_thread.start()

    try:
        agent_main()
    finally:
        # drone.stop_frame_thread()
        keepalive_thread.join()
        capture_thread.join()
        print("All threads closed safely.")


if __name__ == "__main__":
    main()

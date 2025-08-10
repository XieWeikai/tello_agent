from djitellopy import Tello


class Drone:
    def __init__(self):
        self.drone = Tello()
        self.drone.connect()
        
        self.drone.streamoff()
        self.drone.streamon()
        self.frame_reader = self.drone.get_frame_read()
        
    def get_frame(self):
        return self.frame_reader.frame
    


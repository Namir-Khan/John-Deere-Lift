import threading
import time
import cv2
import serial
from ultralytics import YOLO

off_command = [0xFF, 0x0F, 0x00, 0x00, 0x00, 0x08, 0x01, 0x00, 0x70, 0x5D]  
on_command = [0xFF, 0x0F, 0x00, 0x00, 0x00, 0x08, 0x01, 0xFF, 0x30, 0x1D]  

controller_event = threading.Event()
controller_event.set()

relay_state_lock = threading.Lock()
relay_on = False
camera_detections = [False, False]

class Camera:
    def __init__(self, cam_ip):
        self.cap = cv2.VideoCapture(cam_ip)
        self.warm_up()

    def warm_up(self):
        for _ in range(5):
            self.cap.read()

    def capture_frame(self):
        self.cap.grab()
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could Not Capture Image")
        return frame

    def release(self):
        self.cap.release()

class SerialCommunicator:
    def __init__(self, com_no):
        self.ser = serial.Serial(port=com_no, baudrate=9600, parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, timeout=1)

    def send_command(self, command):
        command_bytes = bytes(command)
        self.ser.write(command_bytes)

    def close(self):
        self.ser.close()

class PersonDetector:
    def __init__(self, model_path):
        self.yolo_model = YOLO(model_path)

    def predict_person(self, frame):
        results = self.yolo_model(frame, conf=0.4, save=False, show=False, verbose=False)
        num_person = results[0].boxes.shape
        detection_occurred = num_person[0] > 0
        total_persons_detected = num_person[0]
        return detection_occurred, total_persons_detected

    def validate_prediction(self, num_valid, camera):
        d_count = 0
        for _ in range(num_valid):
            frame = camera.capture_frame()
            detection, total_persons_detected = self.predict_person(frame)
            if detection:
                d_count += 1
        detection_occurred = d_count > (num_valid // 2)
        return detection_occurred, total_persons_detected

class SystemController:
    def __init__(self, model_path, num_valid, cam_ip, cam_num, serial_communicator):
        self.camera = Camera(cam_ip)
        self.serial_communicator = serial_communicator
        self.person_detector = PersonDetector(model_path)
        self.num_valid = num_valid
        self.cam_num = cam_num

    def run(self):
        global relay_on
        global camera_detections

        # Initialize relay to off state at the start
        with relay_state_lock:
            if not relay_on:
                self.serial_communicator.send_command(off_command)
                print(f"Initial: Camera {self.cam_num}: Relay OFF")

        while controller_event.is_set():
            detection_occurred, _ = self.person_detector.validate_prediction(self.num_valid, self.camera)
            with relay_state_lock:
                camera_detections[self.cam_num - 1] = detection_occurred

                if any(camera_detections) and not relay_on:
                    # print(f"Camera {self.cam_num}: Relay ON")
                    print("Relay ON")
                    self.serial_communicator.send_command(on_command)
                    relay_on = True
                elif not any(camera_detections) and relay_on:
                    # print(f"Camera {self.cam_num}: Relay OFF")
                    print("Relay OFF")
                    self.serial_communicator.send_command(off_command)
                    relay_on = False

            time.sleep(0.1)

    def cleanup(self):
        self.camera.release()

def main():
    camera_list = ['rtsp://root:root@192.168.0.92/axis-media/media.amp', 'rtsp://root:root@192.168.0.93/axis-media/media.amp']
    serial_communicator = SerialCommunicator("COM10")

    system_controller1 = SystemController("model_v3_136e.pt", 3, camera_list[0], 1, serial_communicator)
    system_controller2 = SystemController("model_v3_136e.pt", 3, camera_list[1], 2, serial_communicator)

    def run_system_controller(controller):
        try:
            while controller_event.is_set():
                controller.run()
        except Exception as e:
            print(f"Error running system controller: {e}")
        finally:
            controller.cleanup()

    thread1 = threading.Thread(target=run_system_controller, args=(system_controller1,))
    thread2 = threading.Thread(target=run_system_controller, args=(system_controller2,))

    thread1.daemon = True
    thread2.daemon = True

    thread1.start()
    thread2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting Program....")
        controller_event.clear()
        thread1.join()
        thread2.join()
        serial_communicator.close()

if __name__ == "__main__":
    main()

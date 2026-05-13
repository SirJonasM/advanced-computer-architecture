import RPi.GPIO as GPIO
from rpi_ws281x import Adafruit_NeoPixel, Color
import torch
from torchvision import models, transforms
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
import ast
import time
from CameraServerClass import CameraServer
from TRSensors import TRSensors
from ServoControllerClass import ServoController
from default_values import *

class AlphaBot2(object):
    def __init__(self, kp, ki, kd, speed):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.speed = speed        

        self.obstacle_count = 0

        self.AIN1 = 12
        self.AIN2 = 13
        self.BIN1 = 20
        self.BIN2 = 21
        self.ENA = 6
        self.ENB = 26
        self.PA = 25
        self.PB = 25
        self.integral = 0
        self.last_proportional = 0
        self.maximum = 25
        self.DR = 16
        self.DL = 19
        self.CS = 5
        self.Clock = 25
        self.Address = 24
        self.DataOut = 23
        self.Buzzer = 4
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        motor_pins = [self.AIN1, self.AIN2, self.BIN1, self.BIN2, self.ENA, self.ENB]
        for pin in motor_pins:
            GPIO.setup(pin, GPIO.OUT)
        GPIO.setup(self.Clock, GPIO.OUT)
        GPIO.setup(self.CS, GPIO.OUT)
        GPIO.setup(self.Address, GPIO.OUT)
        GPIO.setup(self.DataOut, GPIO.IN, GPIO.PUD_UP)
        GPIO.setup(self.Buzzer, GPIO.OUT)
        GPIO.setup(self.DR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.DL, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.PWMA = GPIO.PWM(self.ENA, 500)
        self.PWMB = GPIO.PWM(self.ENB, 500)
        self.PWMA.start(0)
        self.PWMB.start(0)
        self.stop()
        # Initialize distance sensors
        self.DR_status = 1
        self.DL_status = 1
        # Initialize additional components
        self.tr_sensor = TRSensors()
        self.servo = ServoController()
        self.camera_server = CameraServer()
        # LED Strip Initialization
        self.led_strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ,
                                            LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
        self.led_strip.begin()
        # Initialize object recognition model and labels
        self.object_model = None
        self.imagenet_classes = None
        self.load_object_recognition_model()

    def setMotor(self, left, right):
        """
        left/right: -100 to +100
        positive = forward
        negative = backward
        """

        # clamp values
        left = max(-100, min(100, left))
        right = max(-100, min(100, right))

        # LEFT MOTOR
        if left >= 0:
            GPIO.output(self.AIN1, GPIO.LOW)
            GPIO.output(self.AIN2, GPIO.HIGH)
            self.PWMA.ChangeDutyCycle(left)
        else:
            GPIO.output(self.AIN1, GPIO.HIGH)
            GPIO.output(self.AIN2, GPIO.LOW)
            self.PWMA.ChangeDutyCycle(-left)

        # RIGHT MOTOR
        if right >= 0:
            GPIO.output(self.BIN1, GPIO.LOW)
            GPIO.output(self.BIN2, GPIO.HIGH)
            self.PWMB.ChangeDutyCycle(right)
        else:
            GPIO.output(self.BIN1, GPIO.HIGH)
            GPIO.output(self.BIN2, GPIO.LOW)
            self.PWMB.ChangeDutyCycle(-right)

    # SAFE STOP
    def stop(self):
        self.PWMA.ChangeDutyCycle(0)
        self.PWMB.ChangeDutyCycle(0)

        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.LOW)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.LOW)

    # compatibility aliases for old code
    def setPWMA(self, duty):
        self.PWMA.ChangeDutyCycle(max(0, min(100, duty)))

    def setPWMB(self, duty):
        self.PWMB.ChangeDutyCycle(max(0, min(100, duty)))

    def load_object_recognition_model(self):
        try:
            self.object_model = models.quantization.mobilenet_v2(
                weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1,
                quantize=True
            )
            self.object_model.eval()
            with open("imagenet1000_clsidx_to_labels.txt", "r") as f:
                labels_dict = ast.literal_eval(f.read())
                self.imagenet_classes = [labels_dict[i] for i in range(len(labels_dict))]
            print("Object recognition model loaded successfully.")
        except Exception as e:
            print("Error loading object recognition model:", e)
            self.object_model = None
            self.imagenet_classes = None

    def set_led(self, index, r, g, b):
        """Set a single LED's color."""
        if 0 <= index < LED_COUNT:
            self.led_strip.setPixelColor(index, Color(r, g, b))

    def update_leds(self):
        """Update the LED strip to show the current colors."""
        self.led_strip.show()

    def clear_leds(self):
        """Turn off all LEDs."""
        for i in range(LED_COUNT):
            self.led_strip.setPixelColor(i, Color(0, 0, 0))
        self.led_strip.show()

    def set_leds_default(self):
        """Set a default pattern on the LED strip."""
        self.set_led(0, 255, 0, 0)    # Red
        self.set_led(1, 0, 255, 0)    # Green
        self.set_led(2, 0, 0, 255)    # Blue
        self.set_led(3, 255, 255, 0)  # Yellow
        self.update_leds()
        time.sleep(2)
        self.clear_leds()

    def infrared_obstacle_check(self):
        dr = GPIO.input(self.DR) == 0
        dl = GPIO.input(self.DL) == 0
        current_state = dr or dl

        if current_state and not self.prev_obstacle_state:
            self.obstacle_count += 1
            buzz_amount = ((self.obstacle_count - 1) % 3) + 1
            for _ in range(buzz_amount):
                GPIO.output(self.Buzzer, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(self.Buzzer, GPIO.LOW)
                time.sleep(0.1)

        self.prev_obstacle_state = current_state

    def buzzer_on(self):
        GPIO.output(self.Buzzer, GPIO.HIGH)

    def buzzer_off(self):
        GPIO.output(self.Buzzer, GPIO.LOW)

    # Camera and Recognition Methods
    def start_camera(self):
        self.camera_server.start_server()

    def stop_camera(self):
        self.camera_server.stop_server()

    def recognize_object(self):
        if self.object_model is None or self.imagenet_classes is None:
            print("Object recognition model not loaded. Cannot recognize object.")
            return
        if not hasattr(self, 'camera_server') or not hasattr(self.camera_server, 'picam2'):
            print("Camera server or PiCamera2 not initialized for object recognition.")
            return
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),  # Ensure correct input size
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        try:
            with torch.no_grad():
                frame = self.camera_server.picam2.capture_array()
                if frame is None:
                    print("No frame captured for object recognition.")
                    return
                input_tensor = preprocess(frame)
                input_batch = input_tensor.unsqueeze(0)
                output = self.object_model(input_batch)
                probs = output[0].softmax(dim=0)
                top_prob, top_idx = torch.max(probs, dim=0)
                print(f"Object Recognition: {top_prob.item() * 100:.2f}% {self.imagenet_classes[top_idx.item()]}")
                if top_idx.item() in SHOE_INDICES  :       
                    self.set_led(0, 255, 0, 0)  
                    self.set_led(1, 255, 0, 0)  
                    self.set_led(2, 255, 0, 0)  
                elif top_idx.item() in BOTTLE_INDICES:     
                    self.set_led(0, 255, 255, 0)  
                    self.set_led(1, 255, 255, 0)  
                    self.set_led(2, 255, 255, 0)  
                elif top_idx.item() in  MUG_INDICES:     
                    self.set_led(0, 0, 255, 0)  
                    self.set_led(1, 0, 255, 0)  
                    self.set_led(2, 0, 255, 0)  
                self.update_leds()
        except Exception as e:
            print(f"Error during object recognition: {e}")

    def follow_line(self):
        """Perform one iteration of the PID line following."""
        position, sensors = self.tr_sensor.readLine()
        
        proportional = position - CENTER
        derivative = proportional - self.last_proportional
        
        if any(v < 400 for v in sensors):
            self.integral += proportional
        else:
            self.integral = 0
            
        power_diff = (self.kp * proportional) + (self.ki * self.integral) + (self.kd * derivative)
        self.last_proportional = proportional
        if power_diff > 90:
            power_diff = 90
        elif power_diff < -90:
            power_diff = -90

        self.setMotor(self.speed - power_diff, self.speed + power_diff)

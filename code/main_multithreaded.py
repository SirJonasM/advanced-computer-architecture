import RPi.GPIO as GPIO
from rpi_ws281x import Adafruit_NeoPixel, Color
import argparse
import torch
from torchvision import models, transforms
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
import ast
import time
import threading  # Required for multithreading
from CameraServerClass import CameraServer
from TRSensors import TRSensors
from ServoControllerClass import ServoController

# LED strip configuration constants:
LED_COUNT      = 4      # Number of LED pixels.
LED_PIN        = 18     # GPIO pin connected to the pixels (must support PWM!).
LED_FREQ_HZ    = 800000 # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 5      # DMA channel to use for generating signal (try 5)
LED_BRIGHTNESS = 255    # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False  # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0

# Global flag to shutdown
stop_event = False

KP = 0.3
KI = 0.001
KD = 1.5
SPEED = 15

CENTER = 2000  
POWER_DIFF_MAX = 90

class AlphaBot2(object):
    def __init__(self):
        self.AIN1 = 12
        self.AIN2 = 13
        self.BIN1 = 20
        self.BIN2 = 21
        self.ENA = 6
        self.ENB = 26
        self.PA = 25
        self.PB = 25
        self.obstacle_count = 0
        self.prev_obstacle_state = False
        self.start_time = None
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

        if left >= 0:
            GPIO.output(self.AIN1, GPIO.LOW)
            GPIO.output(self.AIN2, GPIO.HIGH)
            self.PWMA.ChangeDutyCycle(left)
        else:
            GPIO.output(self.AIN1, GPIO.HIGH)
            GPIO.output(self.AIN2, GPIO.LOW)
            self.PWMA.ChangeDutyCycle(-left)

        if right >= 0:
            GPIO.output(self.BIN1, GPIO.LOW)
            GPIO.output(self.BIN2, GPIO.HIGH)
            self.PWMB.ChangeDutyCycle(right)
        else:
            GPIO.output(self.BIN1, GPIO.HIGH)
            GPIO.output(self.BIN2, GPIO.LOW)
            self.PWMB.ChangeDutyCycle(-right)

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
        """Check IR sensors and return True if path is blocked."""
        dr = GPIO.input(self.DR) == 0
        dl = GPIO.input(self.DL) == 0
        current_state = dr or dl

        if current_state and not self.prev_obstacle_state:
            self.stop()
            self.obstacle_count += 1
            print(f"OBSTACLE {self.obstacle_count} | STOPPING")
            
            buzz_amount = ((self.obstacle_count - 1) % 3) + 1
            self.buzz_sync(buzz_amount)

        self.prev_obstacle_state = current_state
        return current_state

    def buzz_sync(self, times):
        """Sequential buzzer: blocks execution while buzzing."""
        for _ in range(times):
            self.buzzer_on()
            time.sleep(0.1)
            self.buzzer_off()
            time.sleep(0.1)

    def buzzer_on(self):
        GPIO.output(self.Buzzer, GPIO.HIGH)

    def buzzer_off(self):
        GPIO.output(self.Buzzer, GPIO.LOW)

    def start_camera(self):
        self.camera_server.start_server()

    def stop_camera(self):
        self.camera_server.stop_server()



class AlphaBot2Multithreaded(AlphaBot2): 
    def __init__(self):
        super().__init__()
        self.running = True
        self.lock = threading.Lock() # CRITICAL: Prevents thread collisions on GPIO
        self.integral = 0 # Initialize integral for PID

    def setMotor(self, left, right):
        """Thread-safe motor control."""
        with self.lock:
            # Re-implementing the core logic here to ensure the lock covers everything
            left = max(-100, min(100, left))
            right = max(-100, min(100, right))
            
            GPIO.output(self.AIN1, GPIO.LOW if left >= 0 else GPIO.HIGH)
            GPIO.output(self.AIN2, GPIO.HIGH if left >= 0 else GPIO.LOW)
            self.PWMA.ChangeDutyCycle(abs(left))

            GPIO.output(self.BIN1, GPIO.LOW if right >= 0 else GPIO.HIGH)
            GPIO.output(self.BIN2, GPIO.HIGH if right >= 0 else GPIO.LOW)
            self.PWMB.ChangeDutyCycle(abs(right))

    def update_leds(self):
        """Thread-safe LED update."""
        with self.lock:
            self.led_strip.show()

    def follow_line(self):
        """Improved PID iteration."""
        position, sensors = self.tr_sensor.readLine()
        proportional = position - CENTER
        
        if any(v < 400 for v in sensors):
            self.integral += proportional
        else:
            self.integral = 0 
            
        derivative = proportional - self.last_proportional
        self.last_proportional = proportional

        power_diff = (KP * proportional) + (KI * self.integral) + (KD * derivative)
        
        power_diff = max(-POWER_DIFF_MAX, min(POWER_DIFF_MAX, power_diff))

        self.setMotor(SPEED - power_diff, SPEED + power_diff)

    def line_following_thread(self):
        print("Starting Line Following Thread")
        time.sleep(3) 
        while self.running:
            self.follow_line()
            # CRITICAL: Allow context switching
            time.sleep(0.002)


    def obstacle_buzzer_thread(self):
        """Thread dedicated to IR detection and buzzing."""
        print("Starting Obstacle Thread")
        while self.running:
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
            time.sleep(0.05)

    def recognition_thread(self):
        """Thread dedicated to camera-based object recognition."""
        print("Starting Recognition Thread")

        SHOE_INDICES   = {770, 774, 630 }   
        BOTTLE_INDICES = {440, 737, 898}         
        MUG_INDICES    = {504, 968}             

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

        while self.running:
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
                    if top_prob.item() > 0.6:
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
                time.sleep(0.5) 


    def set_led_sync(self, r, g, b):
        """Thread-safe LED update."""
        for i in range(4):
            self.led_strip.setPixelColor(i, Color(r, g, b))
        self.led_strip.show()

def parse_args():
    # 1. Setup the Argument Parser
    parser = argparse.ArgumentParser(description='AlphaBot2 Line Follower Configuration')
    
    # 2. Add arguments with your current values as defaults
    parser.add_argument('--kp', type=float, default=0.3, help='Proportional gain (default: 0.3)')
    parser.add_argument('--ki', type=float, default=0.001, help='Integral gain (default: 0.001)')
    parser.add_argument('--kd', type=float, default=1.5, help='Derivative gain (default: 1.5)')
    parser.add_argument('--speed', type=int, default=15, help='Base motor speed (default: 15)')

    # 3. Parse the arguments
    args = parser.parse_args()

    # Use the parsed values
    KP = args.kp
    KI = args.ki
    KD = args.kd
    SPEED = args.speed    

def main():
    bot = AlphaBot2Multithreaded()

    bot.set_led(2, 0, 0, 255)    # Blue
    bot.update_leds()
    bot.buzzer_on()
    time.sleep(0.1)
    bot.buzzer_off()
    bot.start_camera()
    print("Camera server started. Visit http://<your_pi_ip>:5000/ in your browser.")

    time.sleep(2)
    bot.clear_leds()
    bot.tr_sensor.calibratedMin = [164, 142, 176, 138, 177]
    bot.tr_sensor.calibratedMax = [971, 973, 975, 970, 978]

    print("Min:", bot.tr_sensor.calibratedMin)
    print("Max:", bot.tr_sensor.calibratedMax)
    
    t1 = threading.Thread(target=bot.line_following_thread, daemon=True)
    t2 = threading.Thread(target=bot.obstacle_buzzer_thread, daemon=True)
    t3 = threading.Thread(target=bot.recognition_thread, daemon=True)

    try:
        t1.start()
        t2.start()
        t3.start()
        
        while True: 
            time.sleep(10)

            
    except KeyboardInterrupt:
        bot.running = False
        t1.join()
        t2.join()
        t3.join()
        bot.stop()
        GPIO.cleanup()
        bot.stop_camera()
        bot.servo.stop()
        bot.clear_leds()
        print("All operations stopped. Exiting program.")


if __name__ == '__main__':
    parse_args()
    main()

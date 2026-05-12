import RPi.GPIO as GPIO
from rpi_ws281x import Adafruit_NeoPixel, Color
import argparse
import torch
from torchvision import models, transforms
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
import time
import threading  
from CameraServerClass import CameraServer
from TRSensors import TRSensors
from ServoControllerClass import ServoController
from AlphaBot2 import AlphaBot2
from default_values import * 

# Global flag to shutdown
stop_event = False
POWER_DIFF_MAX = 90

class AlphaBot2Multithreaded(AlphaBot2): 
    def __init__(self, kp, ki, kd, speed):
        super().__init__(kp=kp, ki=ki, kd=kd, speed=speed)
        self.running = True
        self.lock = threading.Lock() 
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

        power_diff = (self.kp * proportional) + (self.ki * self.integral) + (self.kd * derivative)
        
        power_diff = max(-POWER_DIFF_MAX, min(POWER_DIFF_MAX, power_diff))

        self.setMotor(self.speed - power_diff, self.speed + power_diff)

    def line_following_thread(self):
        print("Starting Line Following Thread")
        time.sleep(3) 
        while self.running:
            self.follow_line()
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
                    if top_prob.item() > 0.1:
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


def main():
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

def parse_args():
    parser = argparse.ArgumentParser(description='AlphaBot2 Line Follower Configuration')
    parser.add_argument('--kp', type=float, default=0.3, help='Proportional gain (default: 0.3)')
    parser.add_argument('--ki', type=float, default=0.001, help='Integral gain (default: 0.001)')
    parser.add_argument('--kd', type=float, default=1.5, help='Derivative gain (default: 1.5)')
    parser.add_argument('--speed', type=int, default=15, help='Base motor speed (default: 15)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    bot = AlphaBot2Multithreaded(kp=args.kp, ki=args.ki, kd=args.kd, speed= args.speed)
    main()

import RPi.GPIO as GPIO
from rpi_ws281x import Adafruit_NeoPixel, Color
import argparse
import torch
from torchvision import models, transforms
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
import ast
import time
from CameraServerClass import CameraServer
from TRSensors import TRSensors
from ServoControllerClass import ServoController
from AlphaBot2 import AlphaBot2

# LED strip configuration constants:
LED_COUNT      = 4      # Number of LED pixels.
LED_PIN        = 18     # GPIO pin connected to the pixels (must support PWM!).
LED_FREQ_HZ    = 800000 # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 5      # DMA channel to use for generating signal (try 5)
LED_BRIGHTNESS = 255    # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False  # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0

# Global flag to shutdown
CENTER = 2000  # Sensor center value

SHOE_INDICES   = {770, 774, 630 }   
BOTTLE_INDICES = {440, 737, 898}         
MUG_INDICES    = {504, 968}              



def main(bot, not_only_line):
    stop_event = False
    bot.set_led(2, 0, 0, 255)    # Blue
    bot.buzzer_on()
    time.sleep(0.1)
    bot.buzzer_off()
    bot.start_camera()
    print("Camera server started. Visit http://<your_pi_ip>:5000/ in your browser.")
    time.sleep(2)
    bot.clear_leds()
    
    bot.tr_sensor.calibratedMin = [210, 193, 218, 184, 247]
    bot.tr_sensor.calibratedMax = [956, 957, 960, 951, 949]

    print("Min:", bot.tr_sensor.calibratedMin)
    print("Max:", bot.tr_sensor.calibratedMax)

    try:
        while not stop_event:
            bot.follow_line()         
            if not_only_line:
                if bot.infrared_obstacle_check():
                    print("Obstacle detected!")

                bot.recognize_object()
                        
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Stopping execution.")
        stop_event = True
    finally:
        bot.stop()
        bot.stop_camera()
        bot.servo.stop()
        GPIO.cleanup()
        print("All operations stopped. Exiting program.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlphaBot2 Line Follower Configuration')
    
    parser.add_argument('--kp', type=float, default=0.3, help='Proportional gain (default: 0.3)')
    parser.add_argument('--ki', type=float, default=0.001, help='Integral gain (default: 0.001)')
    parser.add_argument('--kd', type=float, default=1.5, help='Derivative gain (default: 1.5)')
    parser.add_argument('--speed', type=int, default=15, help='Base motor speed (default: 15)')
    parser.add_argument('--not-only-line', type=bool, default=False, help='Base motor speed (default: 15)')

    args = parser.parse_args()
    bot = AlphaBot2(kp=args.kp, ki=args.ki, kd=args.kd, speed=args.speed)

    main(bot=bot, not_only_line=args.not_only_line)

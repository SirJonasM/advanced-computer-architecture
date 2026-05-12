import RPi.GPIO as GPIO
import argparse
import time
from AlphaBot2 import AlphaBot2

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
    print("Running with not_only_line: ", not_only_line)

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
    parser.add_argument('--not-only-line', type=bool, default=False, help='Activates Line recognition and object detection')

    args = parser.parse_args()
    bot = AlphaBot2(kp=args.kp, ki=args.ki, kd=args.kd, speed=args.speed)

    main(bot=bot, not_only_line=args.not_only_line)

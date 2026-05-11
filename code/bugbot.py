#!/usr/bin/env python3
# AlphaBot2_multithreaded.py
# Homework #2 – Multithreading Solution
# Tasks:
#   1. Continuously follow the black track
#   2. Infrared obstacle detection → buzzer (1x, 2x, 3x)
#   3. Object recognition via camera → RGB LED (shoe=red, bottle=yellow, mug=green)
#   + Round-trip timing

import RPi.GPIO as GPIO
from rpi_ws281x import Adafruit_NeoPixel, Color
import torch
from torchvision import models, transforms
from torchvision.models.quantization import MobileNet_V2_QuantizedWeights
import ast
import time
import threading

from CameraServerClass import CameraServer
from TRSensors import TRSensors
from ServoControllerClass import ServoController

# ─────────────────────────────────────────────
# LED strip configuration
# ─────────────────────────────────────────────
LED_COUNT      = 4
LED_PIN        = 18
LED_FREQ_HZ    = 800000
LED_DMA        = 5
LED_BRIGHTNESS = 255
LED_INVERT     = False
LED_CHANNEL    = 0

# ─────────────────────────────────────────────
# Line-following tuning
# ─────────────────────────────────────────────
KP     = 0.3      # proportional gain
CENTER = 2000     # sensor centre value (5 sensors × 0-4000 scale)
SPEED  = 10       # base motor speed (0-100)

# ─────────────────────────────────────────────
# Object-recognition: ImageNet class indices
# ─────────────────────────────────────────────
# shoe  → 'running shoe' etc.  common indices: 770 (running shoe), 514 (cornet… no)
# Use a broad set so the demo is robust in practice.
SHOE_INDICES   = {770, 517, 514, 509}   # sneaker/running shoe/sandal/loafer
BOTTLE_INDICES = {440, 737, 899}         # bottle / water_bottle / wine_bottle
MUG_INDICES    = {504, 968}              # coffee_mug / cup

# How often (seconds) the recognition thread runs inference
RECOGNITION_INTERVAL = 1.5


class AlphaBot2:
    def __init__(self):
        # ── GPIO ──────────────────────────────
        self.AIN1 = 12
        self.AIN2 = 13
        self.BIN1 = 20
        self.BIN2 = 21
        self.ENA  = 6
        self.ENB  = 26
        self.PA   = 25
        self.PB   = 25
        self.DR   = 16   # right IR obstacle sensor
        self.DL   = 19   # left  IR obstacle sensor
        self.Buzzer = 4

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in [self.AIN1, self.AIN2, self.BIN1, self.BIN2, self.ENA, self.ENB]:
            GPIO.setup(pin, GPIO.OUT)
        GPIO.setup(self.Buzzer, GPIO.OUT)
        GPIO.setup(self.DR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.DL, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        self.PWMA = GPIO.PWM(self.ENA, 500)
        self.PWMB = GPIO.PWM(self.ENB, 500)
        self.PWMA.start(0)
        self.PWMB.start(0)
        self.stop()

        # ── Subsystems ────────────────────────
        self.tr_sensor     = TRSensors()
        self.servo         = ServoController()
        self.camera_server = CameraServer()

        # ── LED strip ─────────────────────────
        self.led_strip = Adafruit_NeoPixel(
            LED_COUNT, LED_PIN, LED_FREQ_HZ,
            LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL
        )
        self.led_strip.begin()
        self._led_lock = threading.Lock()

        # ── Object-recognition model ──────────
        self.object_model    = None
        self.imagenet_classes = None
        self._load_model()

        # ── Shared state (thread-safe) ─────────
        self._stop_event      = threading.Event()   # set → all threads exit
        self._buzzer_lock     = threading.Lock()
        self._obstacle_count  = 0                   # how many objects detected so far
        self._last_obstacle   = False               # previous IR reading (edge detect)

        # ── Line-following internal state ──────
        self.last_proportional = 0

        # ── Round-trip timing ─────────────────
        self._lap_start_time  = None
        self._lap_running     = False

    # ──────────────────────────────────────────
    # Motor helpers
    # ──────────────────────────────────────────
    def setMotor(self, left, right):
        left  = max(-100, min(100, left))
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
        for pin in [self.AIN1, self.AIN2, self.BIN1, self.BIN2]:
            GPIO.output(pin, GPIO.LOW)

    # ──────────────────────────────────────────
    # Buzzer helpers
    # ──────────────────────────────────────────
    def _buzz(self, times: int):
        """Buzz `times` times (non-blocking: called from its own thread)."""
        with self._buzzer_lock:
            for _ in range(times):
                GPIO.output(self.Buzzer, GPIO.HIGH)
                time.sleep(0.15)
                GPIO.output(self.Buzzer, GPIO.LOW)
                time.sleep(0.15)

    # ──────────────────────────────────────────
    # LED helpers  (thread-safe)
    # ──────────────────────────────────────────
    def _set_all_leds(self, r, g, b):
        with self._led_lock:
            for i in range(LED_COUNT):
                self.led_strip.setPixelColor(i, Color(r, g, b))
            self.led_strip.show()

    def clear_leds(self):
        self._set_all_leds(0, 0, 0)

    # ──────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────
    def _load_model(self):
        try:
            self.object_model = models.quantization.mobilenet_v2(
                weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1,
                quantize=True
            )
            self.object_model.eval()
            with open("imagenet1000_clsidx_to_labels.txt", "r") as f:
                labels_dict = ast.literal_eval(f.read())
            self.imagenet_classes = [labels_dict[i] for i in range(len(labels_dict))]
            print("[Model] Loaded successfully.")
        except Exception as e:
            print(f"[Model] ERROR loading model: {e}")
            self.object_model = None

    # ──────────────────────────────────────────
    # THREAD 1 – Line following
    # ──────────────────────────────────────────
    def _thread_line_follow(self):
        """Runs continuously; uses proportional control to stay on the line."""
        print("[LineFollow] Thread started.")
        while not self._stop_event.is_set():
            try:
                position, sensors = self.tr_sensor.readLine()
                proportional   = position - CENTER
                power_diff     = KP * proportional
                self.last_proportional = proportional

                left_speed  = SPEED - power_diff
                right_speed = SPEED + power_diff
                self.setMotor(left_speed, right_speed)
            except Exception as e:
                print(f"[LineFollow] Error: {e}")
            # No sleep here – tight loop for smooth tracking

        print("[LineFollow] Thread stopped.")

    # ──────────────────────────────────────────
    # THREAD 2 – Infrared obstacle detection + buzzer
    # ──────────────────────────────────────────
    def _thread_obstacle_detection(self):
        """
        Rising-edge detection on the IR sensors.
        Each new obstacle (transition from 'no obstacle' → 'obstacle') increments
        the counter and triggers the buzzer accordingly (1×, 2×, 3×, then resets).
        """
        print("[Obstacle] Thread started.")
        prev_state = False   # False = no obstacle

        while not self._stop_event.is_set():
            dr = GPIO.input(self.DR) == 0   # 0 = obstacle present (active-low)
            dl = GPIO.input(self.DL) == 0
            current_state = dr or dl

            # Rising edge: obstacle just appeared
            if current_state and not prev_state:
                self._obstacle_count += 1
                count = self._obstacle_count

                print(f"[Obstacle] Detected! Total count: {count}")

                # Buzz count times (cycle 1→2→3→1→…)
                buzz_times = ((count - 1) % 3) + 1
                # Run buzzer in its own daemon thread so it doesn't block detection
                t = threading.Thread(
                    target=self._buzz,
                    args=(buzz_times,),
                    daemon=True
                )
                t.start()

            prev_state = current_state
            time.sleep(0.05)   # 50 ms polling is fast enough

        print("[Obstacle] Thread stopped.")

    # ──────────────────────────────────────────
    # THREAD 3 – Object recognition + LED
    # ──────────────────────────────────────────
    def _thread_object_recognition(self):
        """
        Runs inference every RECOGNITION_INTERVAL seconds.
        Maps top-1 prediction to shoe / bottle / mug and sets LED colour.
        """
        print("[Recognition] Thread started.")

        if self.object_model is None:
            print("[Recognition] Model not available – thread exiting.")
            return

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ])

        while not self._stop_event.is_set():
            try:
                frame = self.camera_server.picam2.capture_array()
                if frame is None:
                    time.sleep(RECOGNITION_INTERVAL)
                    continue

                with torch.no_grad():
                    tensor = preprocess(frame).unsqueeze(0)
                    output = self.object_model(tensor)
                    probs  = output[0].softmax(dim=0)
                    top_prob, top_idx = torch.max(probs, dim=0)
                    idx   = top_idx.item()
                    conf  = top_prob.item() * 100
                    label = self.imagenet_classes[idx]

                print(f"[Recognition] {conf:.1f}%  idx={idx}  label={label}")

                if idx in SHOE_INDICES:
                    print("[Recognition] → SHOE  (LED red)")
                    self._set_all_leds(255, 0, 0)
                elif idx in BOTTLE_INDICES:
                    print("[Recognition] → BOTTLE (LED yellow)")
                    self._set_all_leds(255, 255, 0)
                elif idx in MUG_INDICES:
                    print("[Recognition] → MUG  (LED green)")
                    self._set_all_leds(0, 255, 0)
                # else: leave LED as-is (don't flicker for irrelevant classes)

            except Exception as e:
                print(f"[Recognition] Error: {e}")

            time.sleep(RECOGNITION_INTERVAL)

        print("[Recognition] Thread stopped.")

    # ──────────────────────────────────────────
    # Round-trip timer helpers
    # ──────────────────────────────────────────
    def start_lap_timer(self):
        self._lap_start_time = time.time()
        self._lap_running    = True
        print("[Timer] Lap started.")

    def stop_lap_timer(self):
        if self._lap_running and self._lap_start_time is not None:
            elapsed = time.time() - self._lap_start_time
            self._lap_running = False
            print(f"[Timer] Lap time: {elapsed:.2f} s")
            return elapsed
        return None

    # ──────────────────────────────────────────
    # Main run method
    # ──────────────────────────────────────────
    def run(self):
        """Start all threads and block until KeyboardInterrupt."""
        # Start camera server (runs in its own daemon thread internally)
        self.camera_server.start_server()
        print("[Main] Camera server started → http://10.42.0.1:5000")
        time.sleep(2)   # let camera warm up

        # Startup signal
        self._buzz(1)
        self._set_all_leds(0, 0, 255)   # blue = running
        time.sleep(0.5)
        self.clear_leds()

        # Pre-saved calibration values (run calibration once, paste results here)
        self.tr_sensor.calibratedMin = [210, 193, 218, 184, 247]
        self.tr_sensor.calibratedMax = [956, 957, 960, 951, 949]
        print(f"[Main] Calibration Min: {self.tr_sensor.calibratedMin}")
        print(f"[Main] Calibration Max: {self.tr_sensor.calibratedMax}")

        # Start lap timer
        self.start_lap_timer()

        # Create threads
        t_line = threading.Thread(
            target=self._thread_line_follow,
            name="LineFollow",
            daemon=True
        )
        t_obstacle = threading.Thread(
            target=self._thread_obstacle_detection,
            name="ObstacleDetection",
            daemon=True
        )
        t_recognition = threading.Thread(
            target=self._thread_object_recognition,
            name="ObjectRecognition",
            daemon=True
        )

        # Start threads
        t_line.start()
        t_obstacle.start()
        t_recognition.start()

        print("[Main] All threads running. Press CTRL-C to stop.")

        try:
            # Keep main thread alive; join with a timeout so KeyboardInterrupt works
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[Main] KeyboardInterrupt – shutting down…")
        finally:
            self._stop_event.set()   # signal all threads to exit

            # Wait for threads to finish (generous timeout)
            t_line.join(timeout=2)
            t_obstacle.join(timeout=2)
            t_recognition.join(timeout=2)

            lap_time = self.stop_lap_timer()
            if lap_time is not None:
                print(f"[Main] Final lap time: {lap_time:.2f} s")

            self.stop()
            self.camera_server.stop_server()
            self.servo.stop()
            self.clear_leds()
            GPIO.cleanup()
            print("[Main] Cleanup complete. Bye!")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    bot = AlphaBot2()
    bot.run()

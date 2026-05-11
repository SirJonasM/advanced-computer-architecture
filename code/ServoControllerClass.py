from PCA9685Class import PCA9685
import asyncio

class ServoController:
    def __init__(self):
        self.pwm = PCA9685(0x40)
        self.pwm.setPWMFreq(50)
        self.servo_positions = {0: 1500, 1: 1500}  # channel: current_pulse

    def angle_to_pulse(self, angle):
        return int(1500 + (angle / 90.0) * 1000)

    async def set_servo_smooth(self, channel, target_pulse, duration=1.0, interrupt_check=None):
        current_pulse = self.servo_positions[channel]
        steps = abs(target_pulse - current_pulse)
        direction = 1 if target_pulse > current_pulse else -1
        if steps == 0:
            return
        interval = duration / steps
        pulse = current_pulse
        while pulse != target_pulse:
            if interrupt_check is not None and interrupt_check():
                print("Obstacle detected during servo movement. Interrupting motion.")
                break
            pulse += direction
            self.pwm.setServoPulse(channel, pulse)
            self.servo_positions[channel] = pulse
            await asyncio.sleep(interval)

    
    def move_left(self):
        self.pwm.setServoPulse(0,500)

    def move_right(self):
        self.pwm.setServoPulse(0,2500)

    def center(self):
        self.pwm.setServoPulse(0,1500)

    def move_down(self):
        self.pwm.setServoPulse(1,2500)

    def move_up(self):
        self.pwm.setServoPulse(1,500)

    def middle(self):
        self.pwm.setServoPulse(1,1250)
    
    # Async versions of movement methods
    async def move_left_smooth(self, interrupt_check=None):
        await self.set_servo_smooth(0, 500, interrupt_check=interrupt_check)

    async def move_right_smooth(self, interrupt_check=None):
        await self.set_servo_smooth(0, 2500, interrupt_check=interrupt_check)

    async def center_smooth(self, interrupt_check=None):
        await self.set_servo_smooth(0, 1500, interrupt_check=interrupt_check)

    async def move_down_smooth(self, interrupt_check=None):
        await self.set_servo_smooth(1, 2500, interrupt_check=interrupt_check)

    async def move_up_smooth(self, interrupt_check=None):
        await self.set_servo_smooth(1, 500, interrupt_check=interrupt_check)

    async def middle_smooth(self, interrupt_check=None):
        await self.set_servo_smooth(1, 1500, interrupt_check=interrupt_check)

    def stop(self):
        self.pwm.setPWM(0, 0, 0)
        self.pwm.setPWM(1, 0, 0)
        print("Servos stopped and PWM signals disabled")

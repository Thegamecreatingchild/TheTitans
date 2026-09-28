from gpiozero import OutputDevice
from time import sleep

class Solenoid:
    def __init__(self, GPIO: int, delay: float):
        self.GPIO = GPIO
        self.delay = delay
        self.relay = OutputDevice(GPIO, active_high=False, initial_value=False)

    def kick(self):
        try:
            self.relay.on()
            sleep(self.delay) # pulse duration
            self.relay.off()
        finally:
            self.relay.off # guarantee OFF even on crash/Ctrl+C

if __name__ == '__main__':
    kicker = Solenoid(17, 0.01)
    kicker.kick()
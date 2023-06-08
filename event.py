import RPi.GPIO as GPIO
import time

def led(pin, t):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT)

    GPIO.output(pin, True)
    time.sleep(t)

    GPIO.cleanup(pin)

def sound(pin, t):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT)

    GPIO.output(pin, True)
    time.sleep(t)

    GPIO.cleanup(pin)



def fail(): sound(16, 3)

def success(): led(18, 5)
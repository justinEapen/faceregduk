import RPi.GPIO as GPIO
import time

RELAY_PIN = 17  # Change this to the correct GPIO pin you're using

GPIO.setmode(GPIO.BCM)      # Use BCM pin numbering
GPIO.setup(RELAY_PIN, GPIO.OUT)  # Set pin as output

try:
    print("Turning relay ON")
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # or GPIO.LOW depending on relay type
    time.sleep(1)
    print("Turning relay OFF")
    GPIO.output(RELAY_PIN, GPIO.LOW)
    time.sleep(1)
finally:
    GPIO.cleanup()

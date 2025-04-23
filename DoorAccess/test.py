import RPi.GPIO as GPIO
import time

RELAY_PIN = 17  # Replace with your GPIO pin

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH)  # Relay OFF by default

print("Relay should be OFF now.")
time.sleep(3)

print("Turning Relay ON (triggered)...")
GPIO.output(RELAY_PIN, GPIO.LOW)
time.sleep(3)

print("Turning Relay OFF again...")
GPIO.output(RELAY_PIN, GPIO.HIGH)

GPIO.cleanup()

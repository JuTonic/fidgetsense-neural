from dataclasses import dataclass
import serial
import time
import numpy as np
import pyfiglet
from tensorflow.keras.models import load_model
import os

PORT = '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0'
BAUDRATE = 115200
TIMEOUT = 1

ser = serial.Serial(port=PORT, baudrate=BAUDRATE, timeout=TIMEOUT)

model = load_model("./model.keras")

time_diff_arr = []
x1_arr = []
y1_arr = []
z1_arr = []
x2_arr = []
y2_arr = []
z2_arr = []
x3_arr = []
y3_arr = []
z3_arr = []
magnitude_1 = []
magnitude_2 = []
magnitude_3 = []

activity_map = {
    0: "nothing",
    1: "typing",
    2: "scrolling",
    3: "fidgeting"
}

previous_time = 0
with open("readings.csv", "w") as f:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore')
        if line:
            try:
                time, x1, y1, z1, x2, y2, z2, x3, y3, z3 = list(map(lambda x: int(x), line.strip().split(";")))
            except:
                continue
            time_diff_arr.append(time - previous_time)
            previous_time = time
            x1_arr.append(x1)
            x2_arr.append(x2)
            x3_arr.append(x3)
            y1_arr.append(y1) 
            y2_arr.append(y2) 
            y3_arr.append(y3)
            z1_arr.append(z1)
            z2_arr.append(z2)
            z3_arr.append(z3)
            magnitude_1.append(np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2))
            magnitude_2.append(np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2))
            magnitude_3.append(np.sqrt(x3 ** 2 + y3 ** 2 + z3 ** 2))
            
            if len(time_diff_arr) == 250:
                X = np.array([time_diff_arr, x1_arr, y1_arr, z1_arr, x2_arr, y2_arr, z2_arr, x3_arr, y3_arr, z3_arr, magnitude_1, magnitude_2, magnitude_3]).transpose()

                pred = model.predict(np.array([X]))[0]

                pred_activity = pred.argmax(axis=0)

                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"nothing: {pred[0]}; typing: {pred[1]}; scrolling: {pred[2]}; fidgeting: {pred[3]}")
                ascii_art = pyfiglet.figlet_format(activity_map[pred_activity])
                print(ascii_art)

                time_diff_arr = []
                x1_arr = []
                y1_arr = []
                z1_arr = []
                x2_arr = []
                y2_arr = []
                z2_arr = []
                x3_arr = []
                y3_arr = []
                z3_arr = []
                magnitude_1 = []
                magnitude_2 = []
                magnitude_3 = []


#!/usr/bin/env python3
import sys
import time
import signal
from time import sleep
import numpy as np

from pypozyx import (
    PozyxSerial, get_first_pozyx_serial_port, PozyxConstants,
    Coordinates, DeviceCoordinates, SensorData,
    POZYX_SUCCESS
)

# ---------------------------
# Regression
# ---------------------------
terms = [
     2.6857918767599276e-001,
     8.9454936774692606e-001,
    -1.6767880743803291e-005
]

def regress(x):
    t = 1
    r = 0
    for c in terms:
        r += c * t
        t *= x
    return r

# ---------------------------
# Config
# ---------------------------
OFFSET_X = 3325
OFFSET_Y = 831
DEFAULT_REMOTE_TAGS = [0x6800]

ANCHORS = [
    DeviceCoordinates(0x6722, 1, Coordinates(0 - OFFSET_X,     0 - OFFSET_Y, 1109)),
    DeviceCoordinates(0x6772, 1, Coordinates(9210 - OFFSET_X, -1154 - OFFSET_Y, 1637)),
    DeviceCoordinates(0x6764, 1, Coordinates(11591 - OFFSET_X, 8201 - OFFSET_Y, 480)),
    DeviceCoordinates(0x671D, 1, Coordinates(604 - OFFSET_X,   8235 - OFFSET_Y, 1897)),
]

LOOP_INTERVAL = 0.25
PROCESS_NOISE_Q = 100.0
MEASUREMENT_NOISE_R = 40000.0

# ---------------------------
# Kalman 2D
# ---------------------------
class Kalman2D:
    def __init__(self, meas, q, r):
        self.x = np.array([[meas[0]], [meas[1]], [0.0], [0.0]])
        self.P = np.diag([500.0, 500.0, 500.0, 500.0])
        self.q = q
        self.r = r

    def predict(self, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        G = np.array([
            [0.5*dt*dt, 0],
            [0, 0.5*dt*dt],
            [dt, 0],
            [0, dt]
        ])

        Q = G @ (np.eye(2) * self.q) @ G.T

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, meas):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        R = np.eye(2) * self.r
        z = np.array([[meas[0]], [meas[1]]])

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def get_position(self):
        return float(self.x[0]), float(self.x[1])

# ---------------------------
# Helpers
# ---------------------------
def parse_cli_tags():
    args = sys.argv[1:]
    remote_ids = []
    if args:
        for a in args:
            try:
                remote_ids.append(int(a, 16) if str(a).startswith("0x") else int(a))
            except:
                pass
    else:
        remote_ids = DEFAULT_REMOTE_TAGS.copy()
    return [None] + remote_ids

def read_sensors(po, remote_id=None):
    s = SensorData()
    ok = po.getAllSensorData(s, remote_id)
    print(s)
    return s if ok == POZYX_SUCCESS else None

def try_position(po, remote_id=None):
    pos = Coordinates()
    status = po.doPositioning(
        pos,
        PozyxConstants.DIMENSION_2D,
        1000,
        PozyxConstants.POSITIONING_ALGORITHM_UWB_ONLY,
        remote_id=remote_id
    )
    return (True, pos) if status == POZYX_SUCCESS else (False, None)

def human_heading(sensor):
    try:
        h = float(sensor.euler_angles.heading)
        return h - 360 if h > 180 else h
    except:
        return None

# ---------------------------
# Signal
# ---------------------------
running = True
def signal_handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ---------------------------
# MAIN
# ---------------------------
def main():
    tag_ids = parse_cli_tags()

    port = get_first_pozyx_serial_port()
    if port is None:
        print("No Pozyx device found.")
        return

    pozyx = PozyxSerial(port)
    print("Connected:", port)

    # push anchors once
    for t in tag_ids:
        pozyx.clearDevices(t)
        for a in ANCHORS:
            pozyx.addDevice(a, t)

    kalman_filters = {}

    while running:
        ts = time.strftime("%H:%M:%S")

        for t in tag_ids:
            sensor = read_sensors(pozyx, t)
            pos_result = try_position(pozyx, t)

            heading = human_heading(sensor)
            heading_str = f"{heading:.1f}" if heading is not None else "NA"

            if pos_result[0]:
                raw = pos_result[1]
                raw.x = regress(raw.x)

                meas = (float(raw.x), float(raw.y))
                key = t if t else 0

                if key not in kalman_filters:
                    kalman_filters[key] = Kalman2D(
                        meas,
                        PROCESS_NOISE_Q,
                        MEASUREMENT_NOISE_R
                    )

                kf = kalman_filters[key]
                kf.predict(LOOP_INTERVAL)
                kf.update(meas)

                fx, fy = kf.get_position()
                x = fx / 1000
                y = fy / 1000
                status = "OK"
            else:
                x, y = 0.0, 0.0
                status = "FAIL"

            tag_str = hex(t) if t else "LOCAL"

            # ✅ FINAL OUTPUT: SINGLE LINE LOG
            print(f"{ts} | {tag_str} | X={x:.2f} | Y={y:.2f} | H={heading_str} | {status}")

        sleep(LOOP_INTERVAL)

    print("Exit.")

if __name__ == "__main__":
    main()

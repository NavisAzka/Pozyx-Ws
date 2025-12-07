#!/usr/bin/env python3
"""
Minimal Pozyx + Kalman + CSV Logger (SPACE triggers logging)
Shows one clean status line (non-blocking, no flicker).
"""

import sys
import time
import csv
import termios
import tty
import select
import requests

import numpy as np

from time import sleep
from pypozyx import (
    PozyxSerial, get_first_pozyx_serial_port, PozyxConstants,
    Coordinates, DeviceCoordinates, SensorData, POZYX_SUCCESS
)

TAG_TARGET = 0x6800
CSV_FILE   = "position_log.csv"
LOOP_DT    = 0.25

OFFSET_X = 3325
OFFSET_Y = 831

ANCHORS = [
    DeviceCoordinates(0x6722, 1, Coordinates(0 - OFFSET_X,     0 - OFFSET_Y,     1109)),
    DeviceCoordinates(0x6772, 1, Coordinates(9210 - OFFSET_X, -1154 - OFFSET_Y,  1637)),
    DeviceCoordinates(0x6764, 1, Coordinates(11591 - OFFSET_X, 8201 - OFFSET_Y,   480)),
    DeviceCoordinates(0x671D, 1, Coordinates(604 - OFFSET_X,   8235 - OFFSET_Y,  1897)),
]

# ---------------------------
# Kalman Filter 2D
# ---------------------------
class Kalman2D:
    def __init__(self, meas, q=100.0, r=40000.0):
        self.x = np.array([[meas[0]],[meas[1]],[0.0],[0.0]])
        self.P = np.diag([500,500,500,500])
        self.q = q
        self.r = r

    def predict(self, dt):
        F = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1,0],
                      [0,0,0,1]])

        G = np.array([[0.5*dt*dt,0],
                      [0,0.5*dt*dt],
                      [dt,0],
                      [0,dt]])

        Q = G @ (np.eye(2)*self.q) @ G.T

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, meas):
        H = np.array([[1,0,0,0],
                      [0,1,0,0]])
        R = np.eye(2)*self.r
        z = np.array([[meas[0]],[meas[1]]])

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4)-K@H) @ self.P

    def get_xy(self):
        return float(self.x[0]), float(self.x[1])

# ---------------------------
# Keyboard helpers (non-blocking)
# ---------------------------
def key_pressed():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return dr

def read_key():
    return sys.stdin.read(1)

# ---------------------------
# Pozyx Helpers
# ---------------------------
def read_sensors(po, rid):
    s = SensorData()
    return s if po.getAllSensorData(s, rid) == POZYX_SUCCESS else None

def get_position(po, rid):
    pos = Coordinates()
    ok = po.doPositioning(pos, PozyxConstants.DIMENSION_2D, 1000,
                          PozyxConstants.POSITIONING_ALGORITHM_UWB_ONLY,
                          remote_id=rid)
    return (ok == POZYX_SUCCESS, pos)

def get_heading(sensor):
    try:
        h = float(sensor.euler_angles.heading)
        return h - 360 if h > 180 else h
    except:
        return 0.0

# ---------------------------
# Dashboard printer
# ---------------------------
def print_status(x, y, heading):
    sys.stdout.write(
        f"\rTAG 0x6800 | X:{x/1000:6.3f} m | Y:{y/1000:6.3f} m | Head:{heading:7.2f}°     "
    )
    sys.stdout.flush()

def get_robot_pose():
    try:
        url = "http://10.42.0.166/reeman/pose"
        r = requests.get(url, timeout=0.2)
        if r.status_code == 200:
            data = r.json()
            return True, float(data["x"]), float(data["y"]), float(data["theta"])
        return False, None, None, None
    except Exception:
        return False, None, None, None

# ---------------------------
# Main
# ---------------------------
def main():

    port = get_first_pozyx_serial_port()
    if not port:
        print("No Pozyx found.")
        return

    pozyx = PozyxSerial(port)
    print("Connected:", port)

    # configure anchors once
    pozyx.clearDevices(TAG_TARGET)
    for a in ANCHORS:
        pozyx.addDevice(a, TAG_TARGET)

    kalman = None

    # CSV header
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "uwb_x_m", "uwb_y_m", "uwb_heading_deg",
            "robot_x_m", "robot_y_m", "robot_theta_rad"
        ])

    # make keyboard non-blocking
    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            ok, pos = get_position(pozyx, TAG_TARGET)

            if ok:
                meas = (float(pos.x), float(pos.y))

                if kalman is None:
                    kalman = Kalman2D(meas)

                kalman.predict(LOOP_DT)
                kalman.update(meas)
                fx, fy = kalman.get_xy()

                sensor = read_sensors(pozyx, TAG_TARGET)
                heading = get_heading(sensor)

                print_status(fx, fy, heading)

                # SPACE → save ONLY TAG 0x6800 + robot pose
                if key_pressed():
                    if read_key() == " ":

                        ok_robot, rx, ry, rtheta = get_robot_pose()
                        if not ok_robot:
                            print("\n[WARN] Robot pose not available — skip logging.")
                            continue

                        # Format angka 2 desimal
                        uwb_x = fx/1000
                        uwb_y = fy/1000
                        robot_x = rx/1000
                        robot_y = ry/1000

                        with open(CSV_FILE, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                time.time(),
                                f"{uwb_x:.2f}",
                                f"{uwb_y:.2f}",
                                f"{heading:.2f}",
                                f"{robot_x * 1000}",
                                f"{robot_y * 1000}",
                                f"{rtheta}",
                            ])

                        print(f"\n[LOG] Saved → UWB({uwb_x:.2f}, {uwb_y:.2f}) | Robot({robot_x:.2f}, {robot_y:.2f}, {rtheta:.2f})")



            sleep(LOOP_DT)

    except KeyboardInterrupt:
        pass

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print("\nExiting.")


if __name__ == "__main__":
    main()

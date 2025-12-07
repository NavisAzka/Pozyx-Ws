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
from datetime import datetime

import numpy as np

from time import sleep
from pypozyx import (
    PozyxSerial, get_first_pozyx_serial_port, PozyxConstants,
    Coordinates, DeviceCoordinates, SensorData, POZYX_SUCCESS
)

terms_kalman_x = [8.8152290454367233e-002,9.4067674077586172e-001]
terms_kalman_y = [4.0593364554404232e-001,1.0015856050836085e+000]
terms_raw_x = [5.6504892746221502e-002,9.5687345521125955e-001]
terms_raw_y = [2.3873536954904839e-001,1.0351674942325759e+000]

def regress(x, terms):
  t = 1
  r = 0
  for c in terms:
    r += c * t
    t *= x
  return r



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
def print_status(raw_x, raw_y, kal_x, kal_y, regress_x, regress_y, regress_raw_x, regress_raw_y, odom_x, odom_y):
    # sys.stdout.write(
    #     f"{datetime.now()}\n"
    #     f"raw : X:{raw_x:6.3f}\tY:{raw_y:6.3f} \n"
    #     f"fil : X:{kal_x:6.3f}\tY:{kal_y:6.3f}    \n"
    #     f"odom: X:{odom_x:6.3f}\tY:{odom_y:6.3f}    \n"
    #     # f"X:{regress_x:6.3f}\tY:{regress_y:6.3f}    \n"
    # )

    sys.stdout.write(
        f"{time.time()}, "
        f"{raw_x:6.3f}, {raw_y:6.3f}, "
        f"{kal_x:6.3f}, {kal_y:6.3f}, "
        f"{regress_x:6.3f}, {regress_y:6.3f}, "
        f"{regress_raw_x:6.3f}, {regress_raw_y:6.3f}, "
        f"{odom_x:6.3f}, {odom_y:6.3f}    \n"
    )
    # sys.stdout.flush()

def get_robot_pose():
    try:
        url = "http://10.7.101.167/reeman/pose"
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
    kalman_regress = None


    # make keyboard non-blocking
    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    
    odom_x = 0
    odom_y = 0

    try:
        while True:
            ok, pos = get_position(pozyx, TAG_TARGET)

            if ok:
                raw_x = float(pos.x) / 1000
                raw_y = float(pos.y) / 1000

                regress_x_raw = regress(raw_x, terms_raw_x)
                regress_y_raw = regress(raw_y, terms_raw_y)
                
                meas = (raw_x, raw_y)
                meas_regress = (regress_x_raw, regress_y_raw)

                if kalman is None:
                    kalman = Kalman2D(meas, 2500, 5000)
                
                if kalman_regress is None:
                    kalman_regress = Kalman2D(meas_regress, 2500, 5000)

                kalman.predict(LOOP_DT)
                kalman.update(meas)

                kalman_regress.predict(LOOP_DT)
                kalman_regress.update(meas_regress)

                fx, fy = kalman.get_xy()
                fx_regress, fy_regress = kalman_regress.get_xy()
                
                regress_x_kalman = regress(fx, terms_kalman_x)
                regress_y_kalman = regress(fy, terms_kalman_y)
                
                sensor = read_sensors(pozyx, TAG_TARGET)
                heading = get_heading(sensor)

                odom = get_robot_pose()

                if (odom[0] == True):
                    odom_x = odom[1]
                    odom_y = odom[2]


                print_status(raw_x, raw_y, fx, fy, regress_x_kalman, regress_y_kalman, fx_regress, fy_regress, odom_x, odom_y)



            sleep(LOOP_DT)

    except KeyboardInterrupt:
        pass

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print("\nExiting.")


if __name__ == "__main__":
    main()

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
import pandas as pd


from time import sleep
from pypozyx import (
    PozyxSerial, get_first_pozyx_serial_port, PozyxConstants,
    Coordinates, DeviceCoordinates, SensorData, POZYX_SUCCESS
)

termsX2_2 = [
    -1.2838861917401223e-001,
     1.0450381308097056e+000
]

def regressX(x):
  t = 1
  r = 0
  for c in termsX2_2:
    r += c * t
    t *= x
  return r

termsY2_2 = [
    -3.9182911446895652e-001,
     9.9020003655176714e-001
]

def regressY(x):
  t = 1
  r = 0
  for c in termsY2_2:
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
def print_status(raw_x, raw_y, kal_x, kal_y, regress_x, regress_y, odom_x, odom_y):
    # sys.stdout.write(
    #     f"{datetime.now()}\n"
    #     f"raw : X:{raw_x:6.3f}\tY:{raw_y:6.3f} \n"
    #     f"fil : X:{kal_x:6.3f}\tY:{kal_y:6.3f}    \n"
    #     f"odom: X:{odom_x:6.3f}\tY:{odom_y:6.3f}    \n"
    #     # f"X:{regress_x:6.3f}\tY:{regress_y:6.3f}    \n"
    # )

    # sys.stdout.write(
    #     f"{time.time()}, "
    #     f"{raw_x:6.3f}, {raw_y:6.3f}, "
    #     f"{kal_x:6.3f}, {kal_y:6.3f}, "
    #     f"{odom_x:6.3f}, {odom_y:6.3f}    \n"
    # )

    print(f"x: {kal_x:6.3f}, y:{kal_y:6.3f},")
    # sys.stdout.flush()
    None

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

    

    print("Scale     :", s)
    print("Rotation  :\n", R)
    print("Translation:", t.flatten())

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
    
    odom_x = 0
    odom_y = 0

    try:
        while True:
            ok, pos = get_position(pozyx, TAG_TARGET)

            if ok:
                raw_x = float(pos.x) / 1000
                raw_y = float(pos.y) / 1000
                meas = (raw_x, raw_y)
                
                
                regress_x = regressX(raw_x)
                regress_y = regressY(raw_y)
                # meas = (regress_x, regress_y)
                

                if kalman is None:
                    kalman = Kalman2D(meas, 2500, 5000)

                kalman.predict(LOOP_DT)
                kalman.update(meas)
                
                fx, fy = kalman.get_xy()
                uwb_umeyama = uwb_to_odom(fx, fy)

                # print(uwb_to_odom(fx, fy));
                # sensor = read_sensors(pozyx, TAG_TARGET)
                # heading = get_heading(sensor)


                odom = get_robot_pose()
                

                if (odom[0] == True):
                    odom_x = odom[1]
                    odom_y = odom[2]

                

                # print("umeyama x:", uwb_umeyama[0], "y:", uwb_umeyama[1])


                print_status(raw_x, raw_y, uwb_umeyama[0], uwb_umeyama[1], regress_x, regress_y, odom_x, odom_y)


            sleep(LOOP_DT)

    except KeyboardInterrupt:
        pass

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print("\nExiting.")


def umeyama_alignment(src, dst):
    """Compute similarity transform that maps src → dst."""
    assert src.shape == dst.shape

    # Means
    mu_src = src.mean(axis=1, keepdims=True)
    mu_dst = dst.mean(axis=1, keepdims=True)

    # Remove means
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Covariance
    Sigma = dst_c @ src_c.T / src.shape[1]

    # SVD
    U, D, Vt = np.linalg.svd(Sigma)

    # Rotation
    S = np.eye(2)
    if np.linalg.det(U @ Vt) < 0:
        S[1,1] = -1

    R = U @ S @ Vt

    # Scale
    var_src = np.sum(src_c**2) / src.shape[1]
    s = np.sum(D * np.diag(S)) / var_src

    # Translation
    t = mu_dst - s * R @ mu_src

    return s, R, t

# ------------------------------------------------------

df = pd.read_csv("log.csv")
# Use kalman-filtered UWB (better than raw)
uwb = df[['kal_x', 'kal_y']].to_numpy().T
odom = df[['odom_x', 'odom_y']].to_numpy().T

# Compute transform
s, R, t = umeyama_alignment(uwb, odom)

# Function to transform live UWB → ODOM frame
def uwb_to_odom(x, y):
    v = np.array([[x],[y]])
    return (s * R @ v + t).flatten()


if __name__ == "__main__":
    main()

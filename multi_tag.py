#!/usr/bin/env python3
"""
multitag_pozyk_dashboard.py (Kalman-enhanced)
Multitag Pozyx monitor — single-screen dashboard (refresh in-place).
- Adds a lightweight per-tag 2D Kalman filter (constant-velocity) applied to UWB positions.
- State units: millimeters (mm) for position, mm/s for velocity so existing display (pos.x/1000) still prints meters.
- Tuning knobs: process_noise_q (process noise on acceleration), measurement_noise_r (measurement noise on position).
"""
import sys
import time
import signal
from time import sleep

import numpy as np

from pypozyx import (
    PozyxSerial, get_first_pozyx_serial_port, PozyxConstants,
    Coordinates, DeviceCoordinates, SingleRegister, SensorData,
    POZYX_SUCCESS
)


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
# Configuration (edit)
# ---------------------------

OFFSET_X = 3325
OFFSET_Y = 831
DEFAULT_REMOTE_TAGS = [0x6800]
ANCHORS = [
    DeviceCoordinates (0x6722, 1, Coordinates( 0 - OFFSET_X, 0 - OFFSET_Y, 1109)),
    DeviceCoordinates (0x6772 , 1, Coordinates( 9210 - OFFSET_X, -1154 - OFFSET_Y, 1637)),
    DeviceCoordinates (0x6764, 1, Coordinates( 11591 - OFFSET_X, 8201 - OFFSET_Y, 480)),
    DeviceCoordinates (0x671D, 1, Coordinates( 604 - OFFSET_X, 8235 - OFFSET_Y, 1897)),
]
LOOP_INTERVAL = 0.25  # seconds

# ---------------------------
# Simple 2D Kalman Filter (Constant Velocity)
# State vector: [x, y, vx, vy] (x,y in mm)
# Measurement: [x, y]
# ---------------------------
class Kalman2D:
    def __init__(self, meas, process_noise_q=1.0, measurement_noise_r=500.0):
        # State initialization: x, y, vx, vy
        self.x = np.array([[meas[0]], [meas[1]], [0.0], [0.0]])
        # Covariance matrix
        self.P = np.diag([500.0, 500.0, 500.0, 500.0])

        # Process and measurement noise (tunable)
        self.q = process_noise_q  # process noise scale (acceleration noise)
        self.r = measurement_noise_r  # measurement noise variance (position)

    def predict(self, dt):
        # State transition matrix
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # Process noise covariance (assuming white accel noise on vx,vy)
        # Construct G and Qc (continuous) -> discrete Q ~ G*Qc*G.T * dt
        G = np.array([
            [0.5*dt*dt, 0],
            [0, 0.5*dt*dt],
            [dt, 0],
            [0, dt]
        ])
        Qc = np.eye(2) * self.q
        Q = G.dot(Qc).dot(G.T)

        self.x = F.dot(self.x)
        self.P = F.dot(self.P).dot(F.T) + Q

    def update(self, meas):
        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        R = np.eye(2) * self.r
        z = np.array([[meas[0]], [meas[1]]])

        y = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        self.x = self.x + K.dot(y)
        I = np.eye(self.P.shape[0])
        self.P = (I - K.dot(H)).dot(self.P)

    def get_position(self):
        return float(self.x[0, 0]), float(self.x[1, 0])

    def get_state(self):
        return self.x.flatten().tolist()

# ---------------------------
# Terminal helpers
# ---------------------------
def supports_ansi():
    try:
        import os
        return os.isatty(sys.stdout.fileno())
    except Exception:
        return False

_ANSI = supports_ansi()

def clear_screen():
    if _ANSI:
        # cursor home + clear screen
        sys.stdout.write("\033[H\033[J")
    else:
        # fallback: print separator lines
        sys.stdout.write("\n" * 10)
    sys.stdout.flush()

def color(txt, code):
    if not _ANSI:
        return txt
    return f"\033[{code}m{txt}\033[0m"

def green(txt): return color(txt, "92")
def red(txt):   return color(txt, "91")
def yellow(txt):return color(txt, "93")
def cyan(txt):  return color(txt, "96")
def bold(txt):  return color(txt, "1")

# ---------------------------
# Pozyx helpers
# ---------------------------
def parse_cli_tags():
    args = sys.argv[1:]
    remote_ids = []
    if args:
        for a in args:
            try:
                remote_ids.append(int(a, 16) if str(a).startswith("0x") else int(a))
            except Exception:
                pass
    else:
        remote_ids = DEFAULT_REMOTE_TAGS.copy()
    return [None] + remote_ids


def human_error(po, remote_id=None):
    err = SingleRegister()
    st = po.getErrorCode(err, remote_id)
    if st == POZYX_SUCCESS:
        try:
            msg = po.getErrorMessage(err)
        except Exception:
            msg = f"raw:{int(err[0])}"
        try:
            return int(err[0]), msg
        except Exception:
            return None, msg
    return None, "could not query error code"


def read_sensors(po, remote_id=None):
    s = SensorData()
    ok = po.getAllSensorData(s, remote_id)
    if ok == POZYX_SUCCESS:
        return s
    return None


def try_position(po, remote_id=None, retries=2, dimension=PozyxConstants.DIMENSION_2D):
    pos = Coordinates()
    last_msg = "unknown"
    for _ in range(retries):
        status = po.doPositioning(pos, dimension, 1000,
                                  PozyxConstants.POSITIONING_ALGORITHM_UWB_ONLY,
                                  remote_id=remote_id)
        if status == POZYX_SUCCESS:
            return True, pos
        err_code, err_msg = human_error(po, remote_id)
        if err_code is None:
            last_msg = err_msg
        else:
            last_msg = f"0x{err_code:02x} -- {err_msg}"
        sleep(0.04)
    return False, last_msg


def human_heading_from_sensor(sensor):
    try:
        heading = float(sensor.euler_angles.heading)
        if heading > 180.0:
            heading = heading - 360.0
        return heading
    except Exception:
        return None

# ---------------------------
# Dashboard renderer
# ---------------------------
def render_dashboard(header, tag_blocks, anchors_info, footer):
    """
    Build a single string representing the whole dashboard.
    tag_blocks: list of strings per tag
    anchors_info: string for anchors
    """
    lines = []
    lines.append(header)
    lines.append("=" * len(header))
    lines.append(anchors_info)
    lines.append("")
    for tb in tag_blocks:
        lines.append(tb)
        lines.append("-" * 60)
    lines.append("")
    lines.append(footer)
    return "\n".join(lines)


def format_tag_block(tag_id, sensor, pos2d_result):
    # Build a clean block for a single tag
    tid = f"TAG {hex(tag_id) if tag_id is not None else 'LOCAL'}"
    block_lines = []
    block_lines.append(bold(tid))
    block_lines.append("")
    if sensor:
        heading = human_heading_from_sensor(sensor)
        block_lines.append(f" IMU:")
        block_lines.append(f"   Accel : {sensor.acceleration.x:7.1f} {sensor.acceleration.y:7.1f} {sensor.acceleration.z:7.1f} (mg)")
        block_lines.append(f"   Gyro  : {sensor.angular_vel.x:7.2f} {sensor.angular_vel.y:7.2f} {sensor.angular_vel.z:7.2f} (deg/s)")
        block_lines.append(f"   Head  : {heading if heading is not None else 'N/A'} deg")
    else:
        block_lines.append(red("   IMU NOT AVAILABLE / REMOTE UNREACHABLE"))

    # Positioning summary
    if pos2d_result is None:
        block_lines.append(yellow("   POSITION: SKIPPED (no anchors configured)"))
    elif isinstance(pos2d_result, tuple) and pos2d_result[0] is True:
        

        block_lines.append(cyan("   POSITION (2D):"))
        pos = pos2d_result[1]
        block_lines.append((f"     X = {pos.x/1000:.3f}:"))
        pos.x = regress(pos.x)


        # pos expected to have x,y,z in millimeters
        block_lines.append(f"     X = {pos.x/1000:.3f} m   Y = {pos.y/1000:.3f} m   Z = {pos.z/1000:.3f} m")
    else:
        # failure message string
        block_lines.append(red(f"   POSITION FAILED: {pos2d_result}"))

    return "\n".join(block_lines)

# ---------------------------
# Main loop & entry
# ---------------------------
running = True

def signal_handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    tag_ids = parse_cli_tags()
    anchors_info = "Anchors configured: " + (str(len(ANCHORS)) if ANCHORS else "0")
    if ANCHORS:
        afi_lines = ["Anchors:"]
        for a in ANCHORS:
            try:
                afi_lines.append(f" - 0x{a.network_id:04x} @ ({a.pos.x},{a.pos.y},{a.pos.z})")
            except Exception:
                afi_lines.append(" - anchor (invalid)")
        anchors_info = "\n".join(afi_lines)

    header = "Pozyx Multitag Dashboard"
    footer_template = "Cycle: {cycle} | Time: {time} | Tags: {ntags} | Anchors: {nanchors}"

    serial_port = get_first_pozyx_serial_port()
    if serial_port is None:
        print(red("No Pozyx device found on serial. Plug in local tag and retry."))
        return

    pozyx = PozyxSerial(serial_port)

    # Print device info once (not part of refresh)
    print("Connected to Pozyx on", serial_port)
    for t in tag_ids:
        try:
            print("\n--- DeviceInfo for", t, "---")
            pozyx.printDeviceInfo(t)
        except Exception as e:
            print(" printDeviceInfo failed:", e)

    # optionally push anchors (once)
    if ANCHORS:
        for t in tag_ids:
            try:
                pozyx.clearDevices(t)
                for a in ANCHORS:
                    pozyx.addDevice(a, t)
                if len(ANCHORS) > 4:
                    pozyx.setSelectionOfAnchors(PozyxConstants.ANCHOR_SELECT_AUTO, len(ANCHORS), remote_id=t)
            except Exception as e:
                print(" push anchors failed for", t, ":", e)

    cycle = 0

    # per-tag Kalman filters
    kalman_filters = {}
    # tuning (adjust as needed) - process noise (accel variance), measurement noise (pos variance)
    PROCESS_NOISE_Q = 100.0    # mm^2/s^4 scale for accel; raise if model underestimates motion
    MEASUREMENT_NOISE_R = 40000.0  # mm^2 (200 mm stddev -> 40000)

    # main refresh loop
    try:
        while running:
            cycle += 1
            # collect per-tag data
            tag_blocks = []
            for t in tag_ids:
                sensor = read_sensors(pozyx, remote_id=t)
                # determine positioning only if anchors >=3
                if len(ANCHORS) >= 3:
                    pos_result = try_position(pozyx, remote_id=t, retries=2, dimension=PozyxConstants.DIMENSION_2D)
                else:
                    pos_result = None

                # If we got a raw position, feed/update the Kalman filter and produce a filtered Coordinates
                filtered_pos = None
                if isinstance(pos_result, tuple) and pos_result[0] is True:
                    raw_pos = pos_result[1]
                    meas = (float(raw_pos.x), float(raw_pos.y))  # mm
                    # create filter if needed
                    key = t if t is not None else 0
                    kf = kalman_filters.get(key)
                    if kf is None:
                        kf = Kalman2D(meas, process_noise_q=PROCESS_NOISE_Q, measurement_noise_r=MEASUREMENT_NOISE_R)
                        kalman_filters[key] = kf
                    # predict & update with dt = LOOP_INTERVAL
                    kf.predict(LOOP_INTERVAL)
                    kf.update(meas)
                    fx, fy = kf.get_position()
                    # build a lightweight object with x/y/z attributes in mm so format_tag_block can use it
                    class _P:
                        pass
                    fp = _P()
                    fp.x = fx
                    fp.y = fy
                    # preserve raw z for display if available
                    try:
                        fp.z = raw_pos.z
                    except Exception:
                        fp.z = 0
                    filtered_pos = (True, fp)
                else:
                    # pass through pos_result (failed or None)
                    filtered_pos = pos_result

                tb = format_tag_block(t, sensor, filtered_pos)
                tag_blocks.append(tb)

            footer = footer_template.format(
                cycle=cycle, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                ntags=len(tag_ids), nanchors=len(ANCHORS)
            )

            # render & clear screen + write
            output = render_dashboard(header, tag_blocks, anchors_info, footer)
            clear_screen()
            sys.stdout.write(output + "\n")
            sys.stdout.flush()

            # controlled sleep to stay responsive
            steps = max(1, int(LOOP_INTERVAL * 10))
            for _ in range(steps):
                if not running:
                    break
                sleep(0.1)

    except Exception as e:
        clear_screen()
        print("Dashboard stopped due to exception:", e)
    finally:
        print("\nExiting. Goodbye.")

if __name__ == "__main__":
    main()

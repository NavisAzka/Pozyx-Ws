#!/usr/bin/env python3
"""
Simplified Pozyx multitag positioning
No ROS, no OSC. Pure Python.
"""

from time import sleep
import math

from pypozyx import (
    PozyxSerial, get_first_pozyx_serial_port, PozyxConstants,
    Coordinates, DeviceCoordinates, SingleRegister, 
    SensorData, POZYX_SUCCESS
)

from pypozyx.definitions.bitmasks import POZYX_INT_MASK_IMU


class SimplePozyx:
    def __init__(self, pozyx, tag_ids, anchors,
                 algorithm=PozyxConstants.POSITIONING_ALGORITHM_UWB_ONLY,
                 dimension=PozyxConstants.DIMENSION_3D,
                 height=1000):

        self.pozyx = pozyx
        self.tag_ids = tag_ids
        self.anchors = anchors
        self.algorithm = algorithm
        self.dimension = dimension
        self.height = height

    # ---------------------------------------------------------
    # CONFIG
    # ---------------------------------------------------------
    def setup(self):
        print("\n=== Pozyx Simple Positioning ===\n")
        self.configure_anchors()

    def configure_anchors(self):
        for tag_id in self.tag_ids:

            self.pozyx.clearDevices(tag_id)

            for anch in self.anchors:
                self.pozyx.addDevice(anch, tag_id)

            if len(self.anchors) > 4:
                self.pozyx.setSelectionOfAnchors(
                    PozyxConstants.ANCHOR_SELECT_AUTO,
                    len(self.anchors), remote_id=tag_id
                )

            print(f"Tag {tag_id}: anchors configured.")

    # ---------------------------------------------------------
    # LOOP
    # ---------------------------------------------------------
    def loop(self):
        for tag in self.tag_ids:
            position = Coordinates()

            status = self.pozyx.doPositioning(
                position,
                self.dimension,
                self.height,
                self.algorithm,
                remote_id=tag
            )

            if status == POZYX_SUCCESS:
                self.print_position(tag, position)
            else:
                self.print_error(tag)

            # --- SENSOR ORIENTATION ---
            sensor_data = SensorData()
            calibration_status = SingleRegister()

            if (tag is not None or
                self.pozyx.checkForFlag(POZYX_INT_MASK_IMU, 0.01) == POZYX_SUCCESS):

                s = self.pozyx.getAllSensorData(sensor_data, tag)
                s &= self.pozyx.getCalibrationStatus(calibration_status, tag)

                if s == POZYX_SUCCESS:
                    self.print_heading(tag, sensor_data)

    # ---------------------------------------------------------
    # PRINT HELPERS
    # ---------------------------------------------------------
    def print_position(self, tag, pos):
        x = pos.x / 1000.0
        y = pos.y / 1000.0
        z = pos.z / 1000.0
        print(f"[TAG {tag}] Position: ({x:.3f}, {y:.3f}, {z:.3f}) m")

    def print_heading(self, tag, sensor):
        heading = sensor.euler_angles.heading
        # convert to ±180 deg
        if heading > 180:
            heading = heading - 360
        print(f"    Heading: {heading:.2f} deg")

    def print_error(self, tag):
        err = SingleRegister()
        self.pozyx.getErrorCode(err, tag)
        print(f"[TAG {tag}] ERROR: {err}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    serial_port = get_first_pozyx_serial_port()
    if serial_port is None:
        print("No Pozyx detected.")
        quit()

    pozyx = PozyxSerial(serial_port)

    # list of tag IDs
    tag_ids = [0x677B,  0x6800]

    # anchors
    OFFSET_X = 3325
    OFFSET_Y = 831

    anchors = [
        DeviceCoordinates (0x6722, 1, Coordinates( 0 - OFFSET_X, 0 - OFFSET_Y, 1109)),
        DeviceCoordinates (0x6772 , 1, Coordinates( 9210 - OFFSET_X, -1154 - OFFSET_Y, 1637)),
        DeviceCoordinates (0x6764, 1, Coordinates( 11591 - OFFSET_X, 8201 - OFFSET_Y, 480)),
        DeviceCoordinates (0x671D, 1, Coordinates( 604 - OFFSET_X, 8235 - OFFSET_Y, 1897)),
    ]

    tracker = SimplePozyx(pozyx, tag_ids, anchors)
    tracker.setup()

    pozyx.printDeviceList()

    print("\nRunning positioning loop...\n")
    while True:
        tracker.loop()
        sleep(0.1)

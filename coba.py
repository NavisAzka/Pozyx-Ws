#!/usr/bin/env python3
from time import sleep
from pypozyx import (
    PozyxSerial, get_first_pozyx_serial_port, PozyxConstants,
    Coordinates, DeviceCoordinates, SingleRegister, SensorData,
    POZYX_SUCCESS
)

def human_error(po, remote_id=None):
    err = SingleRegister()
    st = po.getErrorCode(err, remote_id)
    if st == POZYX_SUCCESS:
        try:
            msg = po.getErrorMessage(err)
        except Exception:
            msg = str(err[0])
        return (err[0], msg)
    return (None, "could not query local error code")

def info_device(po, dev_id):
    print("=== DeviceInfo for", dev_id, "===")
    try:
        po.printDeviceInfo(dev_id)
    except Exception as e:
        print("  printDeviceInfo failed:", e)
    # try getVersion (may fail if remote unreachable)
    try:
        ver = po.getVersion(remote_id=dev_id)
        print("  getVersion:", ver)
    except Exception as e:
        print("  getVersion failed:", e)

def try_position(po, remote_id=None, retries=3):
    from pypozyx import Coordinates
    for i in range(retries):
        pos = Coordinates()
        status = po.doPositioning(pos, PozyxConstants.DIMENSION_3D, 1000,
                                  PozyxConstants.POSITIONING_ALGORITHM_UWB_ONLY,
                                  remote_id=remote_id)
        if status == POZYX_SUCCESS:
            print(f"[{remote_id}] POS ok: x={pos.x/1000:.3f} y={pos.y/1000:.3f} z={pos.z/1000:.3f} m")
            return True
        else:
            err_code, err_msg = human_error(po, remote_id)
            print(f"[{remote_id}] positioning attempt {i+1}/{retries} failed: err 0x{err_code:x} -- {err_msg}")
            sleep(0.1)
    return False

if __name__ == "__main__":
    serial_port = get_first_pozyx_serial_port()
    if serial_port is None:
        print("No Pozyx on serial.")
        raise SystemExit(1)

    po = PozyxSerial(serial_port)
    # sesuaikan list tag kamu
    tag_ids = [None, 0x677B]

    # print info untuk semua tag
    for t in tag_ids:
        info_device(po, t)
        print("")

    # coba positioning dengan retries
    for t in tag_ids:
        print("Trying positioning for", t)
        ok = try_position(po, remote_id=t, retries=4)
        if not ok:
            print("  -> Final: positioning failed for", t)
        print("")

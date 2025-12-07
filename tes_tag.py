from pypozyx import *
from time import sleep

serial_port = get_first_pozyx_serial_port()
pozyx = PozyxSerial(serial_port)

sensor = SensorData()
calib = SingleRegister()

print("\n=== IMU LOOP (CTRL+C to exit) ===")
while True:

    print("=== DEVICE INFO ===")
    pozyx.printDeviceInfo()
    
    # pozyx.getAllSensorData(sensor)
    # pozyx.getCalibrationStatus(calib)

    # print(f"Accel: {sensor.acceleration.x}, {sensor.acceleration.y}, {sensor.acceleration.z}")
    # print(f"Gyro:  {sensor.angular_vel.x}, {sensor.angular_vel.y}, {sensor.angular_vel.z}")
    # print(f"Euler: roll={sensor.euler_angles.roll}, pitch={sensor.euler_angles.pitch}, head={sensor.euler_angles.heading}")
    # print(f"Calibration: {calib[0]}")
    print("---")

    sleep(0.2)

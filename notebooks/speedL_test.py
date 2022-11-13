import time

import numpy as np

from airo_drake.hardware.universal_robots import UR

ip_victor = "10.42.0.162"
victor = UR("victor", np.identity(4), None, None, None, ip_victor)

print(victor.pose)

velocity = np.array([0, 0, 0] + [0, 0, 0])
victor.rtde_control.speedL(velocity, acceleration=victor.LINEAR_ACCELERATION)
time.sleep(4)
victor.rtde_control.speedL(np.zeros(6), acceleration=victor.LINEAR_ACCELERATION)

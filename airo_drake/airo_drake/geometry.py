import numpy as np


def top_down_orientation(gripper_open_direction) -> np.ndarray:
    X = gripper_open_direction / np.linalg.norm(gripper_open_direction)  # np.array([-1, 0, 0])
    Z = np.array([0, 0, -1])
    Y = np.cross(Z, X)
    return np.column_stack([X, Y, Z])

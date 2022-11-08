import numpy as np


def fake_towel_keypoints(towel_length=0.3, towel_width=0.2, height=0.0, dx=0.02, dy=0.05):
    x = towel_width / 2
    y = towel_length / 2
    z = height
    corners = np.array(
        [
            np.array([x, y, z]),
            np.array([x, -y, z]),
            np.array([-x, -y, z]),
            np.array([-x, y, z]),
        ]
    )
    return corners.T


def angle_2D(v0, v1):
    # TODO: document.
    x1, y1, *_ = v0
    x2, y2, *_ = v1
    dot = x1 * x2 + y1 * y2  # dot product between [x1, y1] and [x2, y2]
    det = x1 * y2 - y1 * x2  # determinant
    angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    return angle


def order_keypoints(keypoints):
    """
    orders keypoints according to their angle w.r.t. a frame that is created by translating the world frame to the center of the cloth.
    the first keypoints is the one with the smallest, positive angle and they are sorted counter-clockwise.
    """
    center = np.mean(keypoints, axis=1)
    x_axis = np.array([1, 0])  # .reshape((2,1))
    angles = [angle_2D(x_axis, keypoint - center) for keypoint in keypoints.T]
    angles = [angle % (2 * np.pi) for angle in angles]  # make angles positive from 0 to 2*pi
    keypoints_sorted = keypoints[:, np.argsort(angles)]
    return keypoints_sorted

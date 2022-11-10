from typing import List, Optional

import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation as R

from airo_drake.hardware.base_classes import Gripper, RobotArm


def homogeneous_pose_to_position_and_rotvec(pose: np.ndarray) -> np.ndarray:
    """converts a 4x4 homogeneous pose to [x,y,z, x_rot,y_rot,z_rot]"""
    position = pose[:3, 3]
    rpy = R.from_matrix(pose[:3, :3]).as_rotvec()
    return np.concatenate((position, rpy))


def position_and_rotvec_to_homogeneuos_pose(pose: np.ndarray) -> np.ndarray:
    orientation = R.from_rotvec(pose[3:6]).as_matrix()
    position = pose[0:3]
    pose = np.eye(4)
    pose[:3, :3] = orientation
    pose[:3, -1] = position
    return pose


class UR(RobotArm):
    """Bridge between our control code and the UR RTDE interface."""

    # Default settings.
    LINEAR_SPEED = 0.1  # m/s
    LINEAR_ACCELERATION = 0.4  # m/s^2
    JOINT_SPEED = 0.4  # rad/s
    JOINT_ACCELERATION = 0.8  # rad/s^2
    BLEND_RADIUS = 0.01  # m?
    MIN_SAFE_TCP_TO_BASE_DISTANCE = 0.08  # m

    def __init__(
        self,
        name: str,
        robot_in_world_pose: np.ndarray,
        home_pose: np.ndarray,
        out_of_way_pose: np.ndarray,
        gripper: Optional[Gripper] = None,
        ip: str = None,
    ):
        if ip is None:
            raise ValueError("IP address is required for real hardware.")
        super().__init__(name, robot_in_world_pose, home_pose, out_of_way_pose, gripper)
        self.ip = ip
        self.rtde_control = RTDEControlInterface(ip)
        self.rtde_receive = RTDEReceiveInterface(ip)

    @property
    def pose(self):
        pose = self.rtde_receive.getActualTCPPose()
        print("RECEIVED POSE")
        print(pose)
        print(position_and_rotvec_to_homogeneuos_pose(pose))
        return position_and_rotvec_to_homogeneuos_pose(pose)

    def move_tcp(self, pose_in_world: np.ndarray):
        self.ensure_pose_safe(pose_in_world)
        pose_in_robot = self.world_to_robot @ pose_in_world
        pose = homogeneous_pose_to_position_and_rotvec(pose_in_robot)
        self.rtde_control.moveJ_IK(pose)

    def move_tcp_linear(self, pose_in_world: np.ndarray, speed: float, acceleration: float):
        self.ensure_pose_safe(pose_in_world)
        pose_in_robot = self.world_to_robot @ pose_in_world
        pose = homogeneous_pose_to_position_and_rotvec(pose_in_robot)
        self.rtde_control.moveL(pose, speed, acceleration)

    def move_tcp_linear_path(self, poses_in_world: List[np.ndarray], speed: float, acceleration: float):
        poses = []
        for pose in poses_in_world:
            self.ensure_pose_safe(pose)
            pose_in_robot = self.world_to_robot @ pose
            pose = homogeneous_pose_to_position_and_rotvec(pose_in_robot)
            # TODO CHECK ORDER OF THESE ADDED ELEMENTS! Below is as docs implies.
            pose_extended = np.concatenate([pose, [acceleration, speed, self.BLEND_RADIUS]])
            poses.append(pose_extended)
        self.rtde_control.moveL(poses)

    def set_tcp_target(self, tcp_desired: np.ndarray, time):
        self.rtde_control.servoL(tcp_desired, 0, 0, time, 0.1, 300)

    def ensure_pose_safe(self, pose_in_world):
        if self.is_pose_unsafe(pose_in_world):
            raise ValueError(f"Cannot safely move {self.name} to world pose {pose_in_world}.")

    def is_pose_unsafe(self, pose_in_world) -> bool:
        """Check whether a given pose is very likely unsafe.
        If this method returns true, the planned robot motion not be executed."""
        pose_in_robot = self.world_to_robot @ pose_in_world
        reachable = self.rtde_control.isPoseWithinSafetyLimits(homogeneous_pose_to_position_and_rotvec(pose_in_robot))
        if not reachable:
            return True

        tcp_to_base_distance = np.linalg.norm(pose_in_robot[:3, -1])
        tcp_too_close_to_base = tcp_to_base_distance < self.MIN_SAFE_TCP_TO_BASE_DISTANCE
        if tcp_too_close_to_base:
            return True

        return False

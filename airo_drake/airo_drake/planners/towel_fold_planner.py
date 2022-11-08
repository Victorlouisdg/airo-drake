import numpy as np
from pydrake.all import AngleAxis, PiecewisePolynomial, PiecewisePose, Rgba, RigidTransform, RotationMatrix

from airo_drake.cloth_manipulation.towel import fake_towel_keypoints, order_keypoints
from airo_drake.geometry import top_down_orientation
from airo_drake.planners.planners_base import DualArmPlannerBase
from airo_drake.visualization import VisualizePath


class TowelFoldPlanner(DualArmPlannerBase):
    def __init__(self, plant, meshcat):
        super().__init__(plant, meshcat)
        self.towel_keypoints = fake_towel_keypoints(height=0.0)  # later this should be an inputport
        VisualizePath(meshcat, "towel_keypoints", self.towel_keypoints, closed=True, color=Rgba(0, 1, 1), thickness=10)

    def get_fold_keyposes(self, keypoints):
        left_grasp_pose = None
        ordered_keypoints = order_keypoints(keypoints)

        top_right_corner, top_left_corner, bottom_left_corner, bottom_right_corner = ordered_keypoints.T

        right_edge = top_right_corner - bottom_right_corner

        topdown = RotationMatrix(top_down_orientation(right_edge))
        tilt_angle = 15
        gripper_y = topdown.col(1)
        local_y_rotation = RotationMatrix(AngleAxis(np.deg2rad(tilt_angle), gripper_y))
        grasp_orientation = local_y_rotation @ topdown

        height_offset = np.array([0, 0, 0.01])
        grasp_position_right = bottom_right_corner + height_offset
        grasp_position_left = bottom_left_corner + height_offset
        right_grasp_pose = RigidTransform(grasp_orientation, grasp_position_right)
        left_grasp_pose = RigidTransform(grasp_orientation, grasp_position_left)

        edge_direction = right_edge / np.linalg.norm(right_edge)
        pregrasp_approach_distance = 0.05
        approach_vector = -pregrasp_approach_distance * edge_direction
        approach_vector[2] += 0.05

        left_pregrasp_pose = RigidTransform(left_grasp_pose)
        left_pregrasp_pose.set_translation(left_pregrasp_pose.translation() + approach_vector)

        right_pregrasp_pose = RigidTransform(right_grasp_pose)
        right_pregrasp_pose.set_translation(right_pregrasp_pose.translation() + approach_vector)

        local_y_rotation = RotationMatrix(AngleAxis(np.deg2rad(-tilt_angle), gripper_y))
        release_orientation = local_y_rotation @ topdown
        right_release_position = top_right_corner + height_offset
        left_release_position = top_left_corner + height_offset
        right_release_pose = RigidTransform(release_orientation, right_release_position)
        left_release_pose = RigidTransform(release_orientation, left_release_position)

        fold_height_offset = np.array([0, 0, np.linalg.norm(right_edge) / 2.0])
        left_middle_position = (grasp_position_left + left_release_position) / 2 + fold_height_offset
        right_middle_position = (grasp_position_right + right_release_position) / 2 + fold_height_offset

        left_middle_pose = RigidTransform(topdown, left_middle_position)
        right_middle_pose = RigidTransform(topdown, right_middle_position)

        left_retreat_position = left_release_position + np.array([0, 0, 0.1])
        right_retreat_position = right_release_position + np.array([0, 0, 0.1])
        left_retreat_pose = RigidTransform(topdown, left_retreat_position)
        right_retreat_pose = RigidTransform(topdown, right_retreat_position)

        left_keyposes = {
            "pregrasp": left_pregrasp_pose,
            "grasp": left_grasp_pose,
            "middle": left_middle_pose,
            "release": left_release_pose,
            "retreat": left_retreat_pose,
        }
        right_keyposes = {
            "pregrasp": right_pregrasp_pose,
            "grasp": right_grasp_pose,
            "middle": right_middle_pose,
            "release": right_release_pose,
            "retreat": right_retreat_pose,
        }
        return left_keyposes, right_keyposes

    def get_gripper_trajectories(self, times):
        opened = np.array([self.gripper_max_open_distance / 2.0])
        closed = np.array([0.0])

        gripper_traj = PiecewisePolynomial.FirstOrderHold(
            [times["initial"], times["pregrasp"]], np.hstack([[closed], [opened]])
        )
        gripper_traj.AppendFirstOrderSegment(times["grasp"], closed)
        gripper_traj.AppendFirstOrderSegment(times["release"], closed)
        gripper_traj.AppendFirstOrderSegment(times["retreat"], opened)
        return gripper_traj, gripper_traj

    def Plan(self, context, state):
        left_keyposes, right_keyposes = self.get_fold_keyposes(self.towel_keypoints)
        left_X_G = {"initial": self.left_X_WE_initial, **left_keyposes}
        right_X_G = {"initial": self.right_X_WE_initial, **right_keyposes}
        times = {"initial": 0.0, "pregrasp": 2.0, "grasp": 2.5, "middle": 3.25, "release": 4.0, "retreat": 5.0}

        def FoldingTrajectory(key_poses, times):
            X_G = list(key_poses.values())
            times = list(times.values())
            traj_X_G = PiecewisePose.MakeCubicLinearWithEndLinearVelocity(times, X_G)
            return traj_X_G

        left_gripper_traj, right_gripper_traj = self.get_gripper_trajectories(times)

        self.left_traj_key_poses = left_X_G
        self.left_traj_X_G = FoldingTrajectory(left_X_G, times)
        self.right_traj_key_poses = right_X_G
        self.right_traj_X_G = FoldingTrajectory(right_X_G, times)

        self.left_gripper_traj = left_gripper_traj
        self.right_gripper_traj = right_gripper_traj

        self.UpdatePlanVisualization()

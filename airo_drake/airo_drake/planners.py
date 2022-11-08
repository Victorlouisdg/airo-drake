from pydrake.all import (
    AbstractValue,
    LeafSystem,
    RigidTransform,
    PiecewisePose,
    Rgba,
    PiecewisePolynomial,
    RotationMatrix,
    AngleAxis,
)
import numpy as np
from airo_drake.cloth_manipulation.towel import fake_towel_keypoints, order_keypoints
from airo_drake.geometry import top_down_orientation

from airo_drake.visualization import AddMeshcatTriad, VisualizePath, VisualizePoseTrajectory


class DualArmPlannerBase(LeafSystem):
    def __init__(self, plant, meshcat):
        LeafSystem.__init__(self)

        # State inputs
        self.DeclareAbstractInputPort("left_X_WE_current", AbstractValue.Make(RigidTransform()))
        self.DeclareAbstractInputPort("right_X_WE_current", AbstractValue.Make(RigidTransform()))
        # I believe gripper_state is distance between fingers and its time derivative.
        self.DeclareVectorInputPort("left_gripper_state", 2)
        self.DeclareVectorInputPort("right_gripper_state", 2)

        # Desired state outputs
        self.DeclareAbstractOutputPort(
            "left_X_WE_desired", lambda: AbstractValue.Make(RigidTransform()), self.OutputLeftGripperPose
        )
        self.DeclareAbstractOutputPort(
            "right_X_WE_desired", lambda: AbstractValue.Make(RigidTransform()), self.OutputRightGripperPose
        )
        self.DeclareVectorOutputPort("left_gripper_position_desired", 1, self.OutputLeftGripperPosition)
        self.DeclareVectorOutputPort("right_gripper_position_desired", 1, self.OutputRightGripperPosition)

        self.plant = plant
        self.inital_pose_right = None
        self.inital_pose_left = None
        self.right_traj_X_G = None
        self.meshcat = meshcat
        self.gripper_max_open_distance = 0.107
        self.left_gripper_traj = None

        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Plan)

    def Initialize(self, context, state):
        left_X_WE_current = self.get_input_port(0).Eval(context)
        right_X_WE_current = self.get_input_port(1).Eval(context)
        left_gripper_state_current = self.get_input_port(2).Eval(context)
        right_gripper_state_current = self.get_input_port(3).Eval(context)
        left_gripper_open_distance = left_gripper_state_current[0]
        right_gripper_open_distance = right_gripper_state_current[1]

        self.right_X_WE_initial = right_X_WE_current
        self.left_X_WE_initial = left_X_WE_current

        left_X_G = {"initial": left_X_WE_current, "hold": left_X_WE_current}
        right_X_G = {"initial": right_X_WE_current, "hold": right_X_WE_current}

        # Simply hold the initialization.
        self.left_traj_X_G = PiecewisePose.MakeLinear([0.0, 1.0], list(left_X_G.values()))
        self.left_traj_key_poses = left_X_G
        self.right_traj_X_G = PiecewisePose.MakeLinear([0.0, 1.0], list(right_X_G.values()))
        self.right_traj_key_poses = right_X_G

        self.left_gripper_traj = PiecewisePolynomial.ZeroOrderHold(
            [0.0, 1.0], [[left_gripper_open_distance, left_gripper_open_distance]]
        )
        self.right_gripper_traj = PiecewisePolynomial.ZeroOrderHold(
            [0.0, 1.0], [[right_gripper_open_distance, right_gripper_open_distance]]
        )

    def Plan(self, context, state):
        """Sets the trajectories for the right and left gripper poses and states."""
        # Implement you planning here, e.g. update the trajectories.
        self.UpdatePlanVisualization()

    def UpdatePlanVisualization(self):
        VisualizePoseTrajectory(self.meshcat, "left_trajectory", self.left_traj_X_G, self.left_traj_key_poses)
        VisualizePoseTrajectory(self.meshcat, "right_trajectory", self.right_traj_X_G, self.right_traj_key_poses)

    def OutputLeftGripperPose(self, context, output):
        X_WE = self.left_traj_X_G.GetPose(context.get_time())
        output.set_value(X_WE)

    def OutputRightGripperPose(self, context, output):
        X_WE = self.right_traj_X_G.GetPose(context.get_time())
        output.set_value(X_WE)

    def OutputLeftGripperPosition(self, context, output):
        output.SetFromVector(self.left_gripper_traj.value(context.get_time()))

    def OutputRightGripperPosition(self, context, output):
        output.SetFromVector(self.right_gripper_traj.value(context.get_time()))


class TowelFoldPlanner(DualArmPlannerBase):
    def __init__(self, plant, meshcat):
        super().__init__(plant, meshcat)
        self.towel_keypoints = fake_towel_keypoints(height=0.0)  # later this should be an inputport
        VisualizePath(meshcat, "towel_keypoints", self.towel_keypoints, closed=True, color=Rgba(0, 1, 1), thickness=10)

    def get_fold_keyposes(self, keypoints):
        left_grasp_pose = None
        ordered_keypoints = order_keypoints(keypoints)
        right_edge = ordered_keypoints[:, 0] - ordered_keypoints[:, 3]

        topdown = RotationMatrix(top_down_orientation(right_edge))
        tilt_angle = 30
        gripper_y = topdown.col(1)
        local_y_rotation = RotationMatrix(AngleAxis(np.deg2rad(tilt_angle), gripper_y))
        rotation = local_y_rotation @ topdown
        bottom_right_corner = ordered_keypoints[:, 3]
        bottom_left_corner = ordered_keypoints[:, 2]

        right_grasp_pose = RigidTransform(rotation, bottom_right_corner)
        left_grasp_pose = RigidTransform(rotation, bottom_left_corner)

        left_keyposes = {"pregrasp": left_grasp_pose, "grasp": left_grasp_pose}
        right_keyposes = {"pregrasp": right_grasp_pose, "grasp": right_grasp_pose}

        return left_keyposes, right_keyposes

    def Plan(self, context, state):
        left_keyposes, right_keyposes = self.get_fold_keyposes(self.towel_keypoints)
        left_X_G = {"initial": self.left_X_WE_initial, **left_keyposes}
        right_X_G = {"initial": self.right_X_WE_initial, **right_keyposes}
        times = {"initial": 0.0, "pregrasp": 2.0, "grasp": 2.1}

        def FoldingTrajectory(key_poses, times):
            X_G = list(key_poses.values())
            times = list(times.values())
            traj_X_G = PiecewisePose.MakeLinear(times, X_G)
            return traj_X_G

        self.left_traj_key_poses = left_X_G
        self.left_traj_X_G = FoldingTrajectory(left_X_G, times)
        self.right_traj_key_poses = right_X_G
        self.right_traj_X_G = FoldingTrajectory(right_X_G, times)

        self.UpdatePlanVisualization()

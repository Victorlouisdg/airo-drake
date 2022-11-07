from pydrake.all import AbstractValue, LeafSystem, RigidTransform, PiecewisePose, Rgba, PiecewisePolynomial
import numpy as np
from airo_drake.cloth_manipulation.towel import fake_towel_keypoints, order_keypoints

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
        self.rigjt_gripper_traj = PiecewisePolynomial.ZeroOrderHold(
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
        output.SetFromVector(self.left_gripper_traj.value(context.get_time()))


class TowelFoldPlanner(DualArmPlannerBase):
    def __init__(self, plant, meshcat):
        super().__init__(plant, meshcat)
        self.towel_keypoints = fake_towel_keypoints(height=0.0)  # later this should be an inputport
        VisualizePath(meshcat, "towel_keypoints", self.towel_keypoints, closed=True, color=Rgba(0, 1, 1), thickness=10)

    def get_grasp_poses(self, keypoints):
        left_grasp_pose = None
        ordered_keypoints = order_keypoints(keypoints)

        initial_X_WE = self.right_X_WE_initial
        initial_rotation = initial_X_WE.rotation()

        bottom_right_corner = ordered_keypoints[:, 3]

        right_grasp_pose = RigidTransform(initial_rotation, bottom_right_corner)
        return left_grasp_pose, right_grasp_pose

    def Plan(self, context, state):
        initial_X_WE = self.right_X_WE_initial

        _, right_grasp_pose = self.get_grasp_poses(self.towel_keypoints)

        X_G = {"initial": initial_X_WE, "pregrasp": right_grasp_pose}
        times = {"initial": 0.0, "pregrasp": 2.0}

        self.right_traj_key_poses = X_G
        X_G = list(X_G.values())
        times = list(times.values())
        traj_X_G = PiecewisePose.MakeLinear(times, X_G)
        self.right_traj_X_G = traj_X_G

        self.UpdatePlanVisualization()

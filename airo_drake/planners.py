from pydrake.all import AbstractValue, LeafSystem, RigidTransform, PiecewisePose, Rgba, PiecewisePolynomial
import numpy as np

from airo_drake.visualization import AddMeshcatTriad


class Planner(LeafSystem):
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

    def CalcRightGripperPoseTrajectory(self, context):
        right_X_WE_current = self.get_input_port(1).Eval(context)

        start = RigidTransform(right_X_WE_current)
        start.set_translation(start.translation() + [0, -0.2, 0])
        down = RigidTransform(start)
        down.set_translation(down.translation() + [0, 0, -0.1])
        back = RigidTransform(down)
        back.set_translation(back.translation() + [0, 0.1, 0])
        up = RigidTransform(back)
        up.set_translation(up.translation() + [0, 0, 0.1])

        X_G = {"initial": right_X_WE_current, "start": start, "down": down, "back": back, "up": up}
        times = {"initial": 0.0, "start": 2.0, "down": 3.0, "back": 4.0, "up": 5.0}

        for name, X in X_G.items():
            AddMeshcatTriad(self.meshcat, f"X_G{name}", X_PT=X)

        X_G = list(X_G.values())
        times = list(times.values())
        traj_X_G = PiecewisePose.MakeLinear(times, X_G)
        self.right_traj_X_G = traj_X_G

        traj_p_G = traj_X_G.get_position_trajectory()
        p_G = traj_p_G.vector_values(traj_p_G.get_segment_times())

        starts = p_G[:, :-1]
        ends = p_G[:, 1:]
        self.meshcat.SetLineSegments("p_G", starts, ends, 2.0, rgba=Rgba(1, 0.65, 0))

    def CalculateLeftFingersTrajectory(self, context):
        opened = self.gripper_max_open_distance
        closed = 0.0
        self.left_gripper_traj = PiecewisePolynomial.FirstOrderHold(
            np.array([0.0, 2.0, 4.0, 5.0]), np.array([closed, opened, opened / 2.0, opened]).reshape([1, -1])
        )

    def OutputLeftGripperPose(self, context, output):
        left_X_WE_current = self.get_input_port(0).Eval(context)
        if self.inital_pose_left is None:
            self.inital_pose_left = RigidTransform(left_X_WE_current)
            t = self.inital_pose_left.translation() + np.array([0, -0.2, 0])
            self.inital_pose_left.set_translation(t)
        output.set_value(self.inital_pose_left)
        # output.set_value(RigidTransform([-0.3, -0.3, 0.5]))

    def OutputRightGripperPose(self, context, output):
        if self.right_traj_X_G is None:
            self.CalcRightGripperPoseTrajectory(context)

        if self.right_traj_X_G is not None:
            X_WE = self.right_traj_X_G.GetPose(context.get_time())
            output.set_value(X_WE)
            return

        if self.inital_pose_right is None:
            right_X_WE_current = self.get_input_port(1).Eval(context)
            self.inital_pose_right = RigidTransform(right_X_WE_current)

        output.set_value(self.inital_pose_right)

    def OutputLeftGripperPosition(self, context, output):
        if self.left_gripper_traj is None:
            self.CalculateLeftFingersTrajectory(context)
        output.SetFromVector(self.left_gripper_traj.value(context.get_time()))

    def OutputRightGripperPosition(self, context, output):
        output.set_value(np.array([self.gripper_max_open_distance]))

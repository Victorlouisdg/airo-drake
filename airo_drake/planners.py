from pydrake.all import AbstractValue, LeafSystem, RigidTransform, PiecewisePose, Rgba
import numpy as np

from airo_drake.visualization import AddMeshcatTriad


class Planner(LeafSystem):
    def __init__(self, plant, meshcat):
        LeafSystem.__init__(self)
        self.DeclareAbstractInputPort("left_X_WE_current", AbstractValue.Make(RigidTransform()))
        self.DeclareAbstractInputPort("right_X_WE_current", AbstractValue.Make(RigidTransform()))

        self.DeclareAbstractOutputPort(
            "left_X_WE_desired", lambda: AbstractValue.Make(RigidTransform()), self.CalcLeftGripperPose
        )
        self.DeclareAbstractOutputPort(
            "right_X_WE_desired", lambda: AbstractValue.Make(RigidTransform()), self.CalcRightGripperPose
        )
        self.plant = plant
        self.inital_pose_right = None
        self.inital_pose_left = None
        self.right_traj_X_G = None
        self.meshcat = meshcat

    def CalcRightGripperTrajectory(self, context):
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

    def CalcLeftGripperPose(self, context, output):
        left_X_WE_current = self.get_input_port(0).Eval(context)
        if self.inital_pose_left is None:
            self.inital_pose_left = RigidTransform(left_X_WE_current)
            t = self.inital_pose_left.translation() + np.array([0, -0.2, 0])
            self.inital_pose_left.set_translation(t)
        output.set_value(self.inital_pose_left)
        # output.set_value(RigidTransform([-0.3, -0.3, 0.5]))

    def CalcRightGripperPose(self, context, output):
        if self.right_traj_X_G is None:
            self.CalcRightGripperTrajectory(context)

        if self.right_traj_X_G is not None:
            X_WE = self.right_traj_X_G.GetPose(context.get_time())
            output.set_value(X_WE)
            return

        if self.inital_pose_right is None:
            right_X_WE_current = self.get_input_port(1).Eval(context)
            self.inital_pose_right = RigidTransform(right_X_WE_current)

        output.set_value(self.inital_pose_right)

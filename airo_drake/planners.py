from pydrake.all import (
    AbstractValue,
    LeafSystem,
    RigidTransform,
)
import numpy as np

class Planner(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.DeclareAbstractInputPort("left_X_WE_current", AbstractValue.Make(RigidTransform()))
        self.DeclareAbstractInputPort("right_X_WE_current", AbstractValue.Make(RigidTransform()))

        self.DeclareAbstractOutputPort("left_X_WE_desired", lambda: AbstractValue.Make(RigidTransform()), self.CalcLeftGripperPose)
        self.DeclareAbstractOutputPort("right_X_WE_desired", lambda: AbstractValue.Make(RigidTransform()), self.CalcRightGripperPose)
        self.plant = plant
        self.inital_pose_right = None
        self.inital_pose_left = None

    def CalcLeftGripperPose(self, context, output):
        left_X_WE_current = self.get_input_port(0).Eval(context)
        if self.inital_pose_left is None:
            self.inital_pose_left = RigidTransform(left_X_WE_current)
            t = self.inital_pose_left.translation() + np.array([0, -0.2, 0])
            self.inital_pose_left.set_translation(t)
        output.set_value(self.inital_pose_left)
        # output.set_value(RigidTransform([-0.3, -0.3, 0.5]))

    def CalcRightGripperPose(self, context, output):
        right_X_WE_current = self.get_input_port(1).Eval(context)
        if self.inital_pose_right is None:
            self.inital_pose_right = RigidTransform(right_X_WE_current)
        output.set_value(self.inital_pose_right)
        # output.set_value(RigidTransform([-0.3, -0.3, 0.5]))


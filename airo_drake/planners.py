from pydrake.all import (
    AbstractValue,
    LeafSystem,
    RigidTransform,
)


class Planner(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.plant = plant
        self.DeclareAbstractOutputPort("left_X_WE", lambda: AbstractValue.Make(RigidTransform()), self.CalcLeftGripperPose)
        self.DeclareAbstractOutputPort("right_X_WE", lambda: AbstractValue.Make(RigidTransform()), self.CalcRightGripperPose)

    def CalcLeftGripperPose(self, context, output):
        plant = self.plant
        output.set_value(RigidTransform([-0.3, -0.3, 0.55]))

    def CalcRightGripperPose(self, context, output):
        output.set_value(RigidTransform([0.3, -0.3, 0.55]))

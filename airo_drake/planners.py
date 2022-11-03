from pydrake.all import (
    AbstractValue,
    LeafSystem,
    RigidTransform,
)


class Planner(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.DeclareAbstractOutputPort("X_WG", lambda: AbstractValue.Make(RigidTransform()), self.CalcGripperPose)

    def CalcGripperPose(self, context, output):
        output.set_value(RigidTransform([0.3, 0, 0.4]))

from pydrake.all import AbstractValue, LeafSystem, RigidTransform


# Adapted from the version in the manipulation repo
class ExtractBodyPose(LeafSystem):
    def __init__(self, plant, body_index):
        LeafSystem.__init__(self)
        self.body_index = body_index
        self.DeclareAbstractInputPort("poses", plant.get_body_poses_output_port().Allocate())
        self.DeclareAbstractOutputPort("pose", lambda: AbstractValue.Make(RigidTransform()), self.CalcOutput)

    def CalcOutput(self, context, output):
        poses = self.EvalAbstractInput(context, 0).get_value()
        pose = poses[int(self.body_index)]
        output.get_mutable_value().set(pose.rotation(), pose.translation())


class ExtractTCPPose(LeafSystem):
    def __init__(self, plant, body_index, tcp_transform):
        LeafSystem.__init__(self)
        self.body_index = body_index
        self.DeclareAbstractInputPort("poses", plant.get_body_poses_output_port().Allocate())
        self.DeclareAbstractOutputPort("pose", lambda: AbstractValue.Make(RigidTransform()), self.CalcOutput)
        self.X_ET = tcp_transform  # RigidTransform([0,0,tcp_offset])

    def CalcOutput(self, context, output):
        poses = self.EvalAbstractInput(context, 0).get_value()
        X_WE = poses[int(self.body_index)]
        X_WT = X_WE @ self.X_ET
        output.get_mutable_value().set(X_WT.rotation(), X_WT.translation())


class WorldToRobotFrame(LeafSystem):
    """Does not work for mobile robots."""

    def __init__(self, plant, model_instance, base_frame):
        LeafSystem.__init__(self)
        self._X_WP_index = self.DeclareAbstractInputPort("X_WP", AbstractValue.Make(RigidTransform()))
        self.DeclareAbstractOutputPort(
            "X_RP", lambda: AbstractValue.Make(RigidTransform()), self.TransformWorldToRobot
        )
        plant_context = plant.CreateDefaultContext()
        X_WR = plant.GetFrameByName(base_frame, model_instance).CalcPoseInWorld(plant_context)
        self.X_RW = X_WR.inverse()

    def TransformWorldToRobot(self, context, output):
        X_WP = self.get_input_port(0).Eval(context)
        X_RP = self.X_RW @ X_WP
        output.set_value(X_RP)


class WorldTCPToRobotEEFFrame(LeafSystem):
    """Does not work for mobile robots.

    X_WT, X_ET, X_RE, X_RW

    X_RE = X_RW @ X_WT @ X_TE
    """

    def __init__(self, plant, model_instance, base_frame, tcp_transform):
        LeafSystem.__init__(self)
        self._X_WT_index = self.DeclareAbstractInputPort("X_WT", AbstractValue.Make(RigidTransform()))
        self.DeclareAbstractOutputPort(
            "X_RE", lambda: AbstractValue.Make(RigidTransform()), self.TransformWorldToRobot
        )
        plant_context = plant.CreateDefaultContext()
        X_WR = plant.GetFrameByName(base_frame, model_instance).CalcPoseInWorld(plant_context)
        self.X_RW = X_WR.inverse()
        # X_ET = RigidTransform([0,0,tcp_offset])
        X_ET = tcp_transform
        self.X_TE = X_ET.inverse()

    def TransformWorldToRobot(self, context, output):
        X_WT = self.get_input_port(0).Eval(context)
        X_RP = self.X_RW @ X_WT @ self.X_TE
        output.set_value(X_RP)

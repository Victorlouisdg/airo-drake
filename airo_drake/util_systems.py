import os
import sys
import warnings

import numpy as np
from pydrake.all import (
    AbstractValue,
    Adder,
    AddMultibodyPlantSceneGraph,
    BallRpyJoint,
    BaseField,
    Box,
    CameraInfo,
    ClippingRange,
    CoulombFriction,
    Cylinder,
    Demultiplexer,
    DepthImageToPointCloud,
    DepthRange,
    DepthRenderCamera,
    DiagramBuilder,
    FindResourceOrThrow,
    GeometryInstance,
    InverseDynamicsController,
    LeafSystem,
    LoadModelDirectivesFromString,
    MakeMultibodyStateToWsgStateSystem,
    MakePhongIllustrationProperties,
    MakeRenderEngineVtk,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    PassThrough,
    PrismaticJoint,
    ProcessModelDirectives,
    RenderCameraCore,
    RenderEngineVtkParams,
    RevoluteJoint,
    Rgba,
    RgbdSensor,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    SchunkWsgPositionController,
    SpatialInertia,
    Sphere,
    StateInterpolatorWithDiscreteDerivative,
    UnitInertia,
    System,
    ConstantVectorSource,
)
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
)

from pydrake.common import GetDrakePath

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

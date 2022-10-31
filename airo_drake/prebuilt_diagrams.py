# Import some basic libraries and functions for this tutorial.
import numpy as np
import os
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.all import (
    InverseDynamicsController,
    PiecewisePose,
    Rgba,
    LeafSystem,
    JacobianWrtVariable,
    TrajectorySource,
    PassThrough,
    Demultiplexer,
    RevoluteJoint,
    StateInterpolatorWithDiscreteDerivative,
)
from pydrake.systems.framework import Diagram, DiagramBuilder, System
from pydrake.common import FindResourceOrThrow, temp_directory
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant

# This file is based on the ManipulationStation used for the MIT Manipulation course

UR_BASE_FRAME = "ur_base_link"


def add_inverse_dynamics_controller(builder, plant, filename, model, num_robot_positions, time_step) -> System:
    controller_plant = MultibodyPlant(time_step=time_step)
    controller_parser = Parser(controller_plant)
    controller_robot_model = controller_parser.AddModelFromFile(filename)
    q0 = np.array([0, -np.pi / 2, 0, -np.pi / 2 + np.pi / 64, 0, 0])
    index = 0
    for joint_index in controller_plant.GetJointIndices(controller_robot_model):
        joint = controller_plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    controller_plant.WeldFrames(controller_plant.world_frame(), controller_plant.GetFrameByName(UR_BASE_FRAME))
    controller_plant.Finalize()

    kp = np.full(num_robot_positions, 100)
    ki = np.full(num_robot_positions, 2 * np.sqrt(kp))
    kd = np.full(num_robot_positions, 20)

    print(kp, kp.shape)

    inverse_dynamics_controller = InverseDynamicsController(controller_plant, kp, ki, kd, False)
    robot_controller = builder.AddSystem(inverse_dynamics_controller)
    robot_controller.set_name("robot_controller")
    builder.Connect(plant.get_state_output_port(model), robot_controller.get_input_port_estimated_state())
    builder.Connect(robot_controller.get_output_port_control(), plant.get_actuation_input_port())
    return robot_controller


def make_ur3e() -> Diagram:
    time_step = 0.001

    filename = FindResourceOrThrow("drake/manipulation/models/ur3e/ur3e_cylinders_collision.urdf")
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant, scene_graph)
    model = parser.AddModelFromFile(filename)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(UR_BASE_FRAME))
    plant.Finalize()

    # Not sure what the use of this pass through is
    model_instance = plant.GetModelInstanceByName("ur3e")
    num_robot_positions = plant.num_positions(model_instance)
    robot_position = builder.AddSystem(PassThrough(num_robot_positions))
    builder.ExportInput(robot_position.get_input_port(), "robot_position")
    builder.ExportOutput(robot_position.get_output_port(), "robot_position_commanded")

    print(num_robot_positions)

    # Export the "state" outputs.
    demux = builder.AddSystem(Demultiplexer(2 * num_robot_positions, num_robot_positions))
    builder.Connect(plant.get_state_output_port(model_instance), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), "robot_position_measured")
    builder.ExportOutput(demux.get_output_port(1), "robot_velocity_estimated")
    builder.ExportOutput(plant.get_state_output_port(model_instance), "robot_state_estimated")

    # Add the InverseDynamicsController controller, which takes desired joint states and commands joint torques.
    robot_controller = add_inverse_dynamics_controller(builder, plant, filename, model, num_robot_positions, time_step)

    # Add discrete derivative to command velocities.
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(num_robot_positions, time_step, suppress_initial_transient=True)
    )
    desired_state_from_position.set_name("robot_desired_state_from_position")
    builder.Connect(desired_state_from_position.get_output_port(), robot_controller.get_input_port_desired_state())
    builder.Connect(robot_position.get_output_port(), desired_state_from_position.get_input_port())

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(plant.get_contact_results_output_port(), "contact_results")
    builder.ExportOutput(plant.get_state_output_port(), "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

    diagram = builder.Build()
    diagram.set_name("UR3e")

    return diagram


def make_ur3e_cart() -> Diagram:
    builder = DiagramBuilder()
    # TODO
    diagram = builder.Build()
    return diagram

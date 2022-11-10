import os

import numpy as np
from pydrake.all import (
    Demultiplexer,
    InverseDynamicsController,
    LoadModelDirectivesFromString,
    MakeMultibodyStateToWsgStateSystem,
    MultibodyPlant,
    Parser,
    PassThrough,
    ProcessModelDirectives,
    RevoluteJoint,
    RigidTransform,
    RotationMatrix,
    SchunkWsgPositionController,
    StateInterpolatorWithDiscreteDerivative,
    System,
)
from pydrake.common import GetDrakePath
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
)

from airo_drake.util_systems import ExtractTCPPose, WorldTCPToRobotEEFFrame


def AddPackagePaths(parser):
    package_directory = os.path.dirname(os.path.dirname(__file__))  # TODO make this more robust
    parser.package_map().PopulateFromFolder(package_directory)
    parser.package_map().Add(
        "manipulation_station", os.path.join(GetDrakePath(), "examples/manipulation_station/models")
    )
    parser.package_map().Add("ycb", os.path.join(GetDrakePath(), "manipulation/models/ycb"))
    parser.package_map().Add(
        "wsg_50_description", os.path.join(GetDrakePath(), "manipulation/models/wsg_50_description")
    )


def CopyModelBetweenPlants(model_instance, plant_from, plant_to):
    print("WARNING: CopyModelBetweenPlants not implemented correctly yet!")
    # TODO Figure out how to implement actual copy.
    name = plant_from.GetModelInstanceName(model_instance)
    # plant_to.AddModelInstance(name)
    # plant_to.some_function()

    # Temporary workaround, reload from URDF.
    # filename = FindResourceOrThrow("drake/manipulation/models/ur3e/ur3e_cylinders_collision.urdf")
    parser = Parser(plant_to)
    AddPackagePaths(parser)

    model_directives = """
    directives:
    - add_directives:
        file: package://airo_drake_models/ur3e_and_wsg.dmd.yaml
    """

    if name == "ur3e":
        model_directives = """
    directives:
    - add_directives:
        file: package://airo_drake_models/ur3e.dmd.yaml
    """

    directives = LoadModelDirectivesFromString(model_directives)
    ProcessModelDirectives(directives, parser)
    model_index = plant_to.GetModelInstanceByName("ur3e")

    # model = parser.AddModelFromFile(filename)
    q0 = plant_from.GetPositions(plant_from.CreateDefaultContext(), model_instance)
    index = 0
    for joint_index in plant_to.GetJointIndices(model_index):
        joint = plant_to.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    # UR_BASE_FRAME = "ur_base_link"
    # plant_to.WeldFrames(plant_to.world_frame(), plant_to.GetFrameByName(UR_BASE_FRAME))


def AddInverseDynamicsController(builder, plant, model_instance) -> System:
    # The InverseDynamicsController needs a plant with only a single robot model and no clutter etc.
    controller_plant = MultibodyPlant(time_step=plant.time_step())
    CopyModelBetweenPlants(model_instance, plant, controller_plant)
    controller_plant.Finalize()

    # TODO someone with PID knowledge tune these gains
    num_robot_positions = plant.num_positions(model_instance)
    kp = np.full(num_robot_positions, 100)
    ki = np.full(num_robot_positions, 100)
    kd = np.full(num_robot_positions, 100)

    inverse_dynamics_controller = InverseDynamicsController(
        controller_plant, kp, ki, kd, has_reference_acceleration=False
    )
    robot_controller = builder.AddSystem(inverse_dynamics_controller)
    builder.Connect(plant.get_state_output_port(model_instance), robot_controller.get_input_port_estimated_state())
    builder.Connect(robot_controller.get_output_port_control(), plant.get_actuation_input_port(model_instance))
    return robot_controller


def AddDifferentialIKIntegrator(builder, dynamics_controller):
    controller_plant = dynamics_controller.get_multibody_plant_for_control()
    params = DifferentialInverseKinematicsParameters(
        controller_plant.num_positions(), controller_plant.num_velocities()
    )
    q0 = controller_plant.GetPositions(controller_plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0)

    # TODO Don't hardcode these limits.
    # These two were arbitrarily chosen.
    params.set_end_effector_angular_speed_limit(3.14)
    params.set_end_effector_translational_velocity_limits([-1, -1, -1], [1, 1, 1])

    # UR3e velocity limits, check whether available from URDF?
    velocity_limits = np.array(3 * [1.57] + 3 * [3.14])
    params.set_joint_velocity_limits((-velocity_limits, velocity_limits))

    num_robot_positions = controller_plant.num_positions()
    params.set_joint_centering_gain(10 * np.eye(num_robot_positions))  # not sure what this is used for

    # Hardcoded name of link in the WSG SDF
    frame = controller_plant.GetFrameByName("ur_tool0")

    differential_IK_integrator = builder.AddSystem(
        DifferentialInverseKinematicsIntegrator(
            controller_plant, frame, controller_plant.time_step(), params, log_only_when_result_state_changes=True
        )
    )
    return differential_IK_integrator


def SetupUR3e(builder, plant, model_instance):
    robot_name = plant.GetModelInstanceName(model_instance)

    num_robot_positions = plant.num_positions(model_instance)
    robot_position = builder.AddSystem(PassThrough(num_robot_positions))
    builder.ExportInput(robot_position.get_input_port(), f"{robot_name}_position")
    builder.ExportOutput(robot_position.get_output_port(), f"{robot_name}_position_commanded")

    # Export the "state" outputs.
    demux = builder.AddSystem(Demultiplexer(2 * num_robot_positions, num_robot_positions))
    builder.Connect(plant.get_state_output_port(model_instance), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), f"{robot_name}_position_measured")
    builder.ExportOutput(demux.get_output_port(1), f"{robot_name}_velocity_estimated")
    builder.ExportOutput(plant.get_state_output_port(model_instance), f"{robot_name}_state_estimated")

    # Add the InverseDynamicsController controller, which takes desired joint states and commands joint torques.
    dynamics_controller = AddInverseDynamicsController(builder, plant, model_instance)
    dynamics_controller.set_name(f"{robot_name}_dynamics_controller")

    # Add the DifferentialIK, which takes gripper poses in robot frame and outputs joint positions
    diff_ik = AddDifferentialIKIntegrator(builder, dynamics_controller)

    tcp_transform = RigidTransform(RotationMatrix.MakeZRotation(np.deg2rad(90)), [0, 0, 0.16])
    # tcp_offset = 0.16
    transform = builder.AddSystem(WorldTCPToRobotEEFFrame(plant, model_instance, "ur_base_link", tcp_transform))
    builder.ExportInput(transform.get_input_port(0), f"{robot_name}_tcp_target")
    builder.Connect(transform.get_output_port(0), diff_ik.get_input_port(0))
    builder.Connect(plant.get_state_output_port(model_instance), diff_ik.GetInputPort("robot_state"))

    # Add discrete derivative to command velocities.
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_robot_positions, plant.time_step(), suppress_initial_transient=True
        )
    )
    desired_state_from_position.set_name(f"{robot_name}_desired_state_from_position")
    builder.Connect(diff_ik.GetOutputPort("joint_positions"), desired_state_from_position.get_input_port())
    builder.Connect(desired_state_from_position.get_output_port(), dynamics_controller.get_input_port_desired_state())

    body_index = plant.GetBodyByName("ur_tool0", model_instance).index()
    gripper_pose = builder.AddSystem(ExtractTCPPose(plant, body_index, tcp_transform))
    builder.Connect(plant.get_body_poses_output_port(), gripper_pose.GetInputPort("poses"))
    builder.ExportOutput(gripper_pose.GetOutputPort("pose"), f"{robot_name}_tcp")


def SetupWSG50(builder, plant, model_instance):
    gripper_name = plant.GetModelInstanceName(model_instance)
    wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    wsg_controller.set_name(gripper_name + "_controller")

    builder.Connect(wsg_controller.get_generalized_force_output_port(), plant.get_actuation_input_port(model_instance))
    builder.Connect(plant.get_state_output_port(model_instance), wsg_controller.get_state_input_port())

    builder.ExportInput(wsg_controller.get_desired_position_input_port(), gripper_name + "_openness_target")

    wsg_mbp_state_to_wsg_state = builder.AddSystem(MakeMultibodyStateToWsgStateSystem())
    builder.Connect(plant.get_state_output_port(model_instance), wsg_mbp_state_to_wsg_state.get_input_port())
    builder.ExportOutput(wsg_mbp_state_to_wsg_state.get_output_port(), gripper_name + "_openness_state")
    builder.ExportOutput(wsg_controller.get_grip_force_output_port(), gripper_name + "_force")


def ExportCheatPorts(builder, scene_graph, plant):
    """Port that can be useful for development in simulation, but aren't available on real hardware."""
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(plant.get_contact_results_output_port(), "contact_results")
    builder.ExportOutput(plant.get_state_output_port(), "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

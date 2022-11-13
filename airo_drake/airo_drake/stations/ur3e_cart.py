from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectivesFromString,
    ModelInstanceIndex,
    Parser,
    ProcessModelDirectives,
)
from pydrake.geometry import MeshcatVisualizer
from pydrake.systems.all import Simulator

from airo_drake.stations.station_utils import AddPackagePaths, ExportCheatPorts, SetupUR3e, SetupWSG50


def AddUR3eCartAndModelsToPlant(plant, additional_directives):
    # TODO eventually make this function more general.
    parser = Parser(plant)
    AddPackagePaths(parser)
    model_directives = (
        """
    directives:
    - add_directives:
        file: package://airo_drake_models/dual_ur3e_and_wsg.dmd.yaml
    """
        + additional_directives
    )
    directives = LoadModelDirectivesFromString(model_directives)
    ProcessModelDirectives(directives, parser)


def MakeUR3eCartStation(additional_directives="", robots_prefix="ur3e", gripper_prefix="wsg", time_step=0.001):
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    AddUR3eCartAndModelsToPlant(plant, additional_directives)
    plant.Finalize()

    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)
        if model_instance_name.startswith(robots_prefix):
            SetupUR3e(builder, plant, model_instance)
        if model_instance_name.startswith(gripper_prefix):
            SetupWSG50(builder, plant, model_instance)

    ExportCheatPorts(builder, scene_graph, plant)

    diagram = builder.Build()
    diagram.set_name("UR3e Cart Station")
    return diagram


def ConnectUR3eCartWithDualArmPlanner(builder, station, planner):
    # Connect state estimation to planner.
    builder.Connect(station.GetOutputPort("ur3e_left_tcp"), planner.GetInputPort("left_tcp"))
    builder.Connect(station.GetOutputPort("ur3e_right_tcp"), planner.GetInputPort("right_tcp"))
    builder.Connect(station.GetOutputPort("wsg_left_openness_state"), planner.GetInputPort("left_openness_state"))
    builder.Connect(station.GetOutputPort("wsg_right_openness_state"), planner.GetInputPort("right_openness_state"))

    # Connect planner output to station.
    builder.Connect(planner.GetOutputPort("left_tcp_desired"), station.GetInputPort("ur3e_left_tcp_target"))
    builder.Connect(planner.GetOutputPort("right_tcp_desired"), station.GetInputPort("ur3e_right_tcp_target"))
    builder.Connect(planner.GetOutputPort("left_openness_desired"), station.GetInputPort("wsg_left_openness_target"))
    builder.Connect(planner.GetOutputPort("right_openness_desired"), station.GetInputPort("wsg_right_openness_target"))


def RunAndPublishSimulation(station, planner, meshcat, simulation_time=6.0):
    builder = DiagramBuilder()
    builder.AddSystem(station)
    builder.AddSystem(planner)
    ConnectUR3eCartWithDualArmPlanner(builder, station, planner)

    visualizer = MeshcatVisualizer.AddToBuilder(builder, station.GetOutputPort("query_object"), meshcat)
    diagram = builder.Build()

    context = diagram.CreateDefaultContext()

    plant = station.GetSubsystemByName("plant")
    body = plant.GetBodyByName("link_0")
    body.SetMass(context, 0.2)

    simulator = Simulator(diagram, context)
    visualizer.StartRecording(False)
    simulator.AdvanceTo(6.0)
    visualizer.PublishRecording()

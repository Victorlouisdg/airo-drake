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

from airo_drake.stations.station_utils import AddPackagePaths, ExportCheatPorts, SetupUR3e


def AddUR3eCartAndModelsToPlant(plant, additional_directives):
    # TODO eventually make this function more general.
    parser = Parser(plant)
    AddPackagePaths(parser)
    model_directives = (
        """
    directives:
    - add_directives:
        file: package://airo_drake_models/ur3e.dmd.yaml
    """
        + additional_directives
    )

    print(model_directives)
    directives = LoadModelDirectivesFromString(model_directives)
    ProcessModelDirectives(directives, parser)


def MakeUR3eCartStation(additional_directives="", robots_prefix="ur3e", time_step=0.001):
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    AddUR3eCartAndModelsToPlant(plant, additional_directives)
    plant.Finalize()

    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)
        if model_instance_name.startswith(robots_prefix):
            SetupUR3e(builder, plant, model_instance, tcp_offset=0.05)

    ExportCheatPorts(builder, scene_graph, plant)

    diagram = builder.Build()
    diagram.set_name("UR3e Station")
    return diagram


def ConnectUR3etWithPlanner(builder, station, planner):
    # Connect state estimation to planner.
    builder.Connect(station.GetOutputPort("ur3e_tcp"), planner.GetInputPort("tcp"))

    # Connect planner output to station.
    builder.Connect(planner.GetOutputPort("tcp_desired"), station.GetInputPort("ur3e_tcp_target"))


def RunAndPublishSimulation(station, planner, meshcat, simulation_time=10.0):
    builder = DiagramBuilder()
    builder.AddSystem(station)
    builder.AddSystem(planner)
    ConnectUR3etWithPlanner(builder, station, planner)

    visualizer = MeshcatVisualizer.AddToBuilder(builder, station.GetOutputPort("query_object"), meshcat)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    visualizer.StartRecording(False)
    simulator.AdvanceTo(simulation_time)
    visualizer.PublishRecording()

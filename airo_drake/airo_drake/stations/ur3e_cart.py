from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder, ModelInstanceIndex
from pydrake.geometry import MeshcatVisualizer
from pydrake.systems.all import Simulator

from airo_drake.stations.station_utils import AddModelsToPlant, SetupUR3e, SetupWSG50


def MakeUR3eCartStation(additional_directives="", robots_prefix="ur3e", gripper_prefix="wsg", time_step=0.001):
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    AddModelsToPlant(plant, additional_directives)
    plant.Finalize()

    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)
        if model_instance_name.startswith(robots_prefix):
            SetupUR3e(builder, plant, model_instance)
        if model_instance_name.startswith(gripper_prefix):
            SetupWSG50(builder, plant, model_instance)

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(plant.get_contact_results_output_port(), "contact_results")
    builder.ExportOutput(plant.get_state_output_port(), "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

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
    simulator = Simulator(diagram)
    visualizer.StartRecording(False)
    simulator.AdvanceTo(6.0)
    visualizer.PublishRecording()

import numpy as np
from pydrake.all import DiagramBuilder, RotationMatrix
from pydrake.systems.all import AbstractValue, LeafSystem, RigidTransform, Simulator

from airo_drake.hardware.universal_robots import UR
from airo_drake.stations.single_ur3e import ConnectUR3etWithPlanner


class UR3eStationHardwareInterface(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareAbstractInputPort("ur3e_tcp_target", AbstractValue.Make(RigidTransform()))
        self.DeclareAbstractOutputPort(
            "ur3e_tcp",
            lambda: AbstractValue.Make(RigidTransform()),
            self.OutputTCP,
            prerequisites_of_calc=set([self.time_ticket()]),
        )

        ip_victor = "10.42.0.162"

        # self.victor = FakeArm("victor", np.identity(4), None, None, None)  # , ip_victor)
        self.victor = UR("victor", np.identity(4), None, None, None, ip_victor)

        self.control_period = 0.05
        self.DeclarePeriodicUnrestrictedUpdateEvent(self.control_period, 0.0, self.CommandTCPPose)

    def CommandTCPPose(self, context, state):
        tcp_desired = self.GetInputPort("ur3e_tcp_target").Eval(context)
        tcp_desired = np.array(tcp_desired.GetAsMatrix4())
        tcp_current = self.victor.pose

        position_desired = tcp_desired[:3, -1]
        position_current = tcp_current[:3, -1]
        distance = np.linalg.norm(position_desired - position_current)
        speed = distance / self.control_period

        self.victor.move_tcp_linear(tcp_desired, speed, self.victor.LINEAR_ACCELERATION)

    def OutputTCP(self, context, output):
        pose = self.victor.pose
        output.set_value(RigidTransform(RotationMatrix(pose[:3, :3]), pose[:3, -1]))


def RunAndPublishOnRealHardware(station, planner, meshcat, simulation_time=6.0):
    builder = DiagramBuilder()
    builder.AddSystem(station)
    builder.AddSystem(planner)
    ConnectUR3etWithPlanner(builder, station, planner)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.AdvanceTo(6.0)

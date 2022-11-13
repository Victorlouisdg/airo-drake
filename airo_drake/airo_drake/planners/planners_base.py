from pydrake.all import AbstractValue, LeafSystem, PiecewisePolynomial, PiecewisePose, RigidTransform

from airo_drake.visualization import VisualizePoseTrajectory


class PlannerBase(LeafSystem):
    def __init__(self, meshcat):
        LeafSystem.__init__(self)
        self.DeclareAbstractInputPort("tcp", AbstractValue.Make(RigidTransform()))
        self.DeclareAbstractOutputPort("tcp_desired", lambda: AbstractValue.Make(RigidTransform()), self.OutputTCP)

        self.inital_tcp = None
        self.tcp_trajectory = None
        self.meshcat = meshcat

        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Plan)

    def Initialize(self, context, state):
        self.inital_tcp = self.GetInputPort("tcp").Eval(context)
        tcp_keyposes = {"initial": self.inital_tcp, "hold": self.inital_tcp}

        # Simply hold the initialization.
        self.tcp_trajectory = PiecewisePose.MakeLinear([0.0, 1.0], list(tcp_keyposes.values()))
        self.tcp_keyposes = tcp_keyposes

    def Plan(self, context, state):
        """Sets the trajectories for the right and left gripper poses and states."""
        # Implement you planning here, e.g. update the trajectories.
        self.UpdatePlanVisualization()

    def UpdatePlanVisualization(self):
        VisualizePoseTrajectory(self.meshcat, "tcp_trajectory", self.tcp_trajectory, self.tcp_keyposes)

    def OutputTCP(self, context, output):
        tcp_pose = self.tcp_trajectory.GetPose(context.get_time())
        output.set_value(tcp_pose)


class DualArmPlannerBase(LeafSystem):
    def __init__(self, meshcat):
        LeafSystem.__init__(self)
        # State inputs
        self.DeclareAbstractInputPort("left_tcp", AbstractValue.Make(RigidTransform()))
        self.DeclareAbstractInputPort("right_tcp", AbstractValue.Make(RigidTransform()))
        # openness_state is distance between fingers and its time derivative.
        self.DeclareVectorInputPort("left_openness_state", 2)
        self.DeclareVectorInputPort("right_openness_state", 2)

        # Desired state outputs
        self.DeclareAbstractOutputPort(
            "left_tcp_desired", lambda: AbstractValue.Make(RigidTransform()), self.OutputLeftTCP
        )
        self.DeclareAbstractOutputPort(
            "right_tcp_desired", lambda: AbstractValue.Make(RigidTransform()), self.OutputRightTCP
        )
        self.DeclareVectorOutputPort("left_openness_desired", 1, self.OutputLeftOpenness)
        self.DeclareVectorOutputPort("right_openness_desired", 1, self.OutputRightOpenness)

        self.gripper_max_openness = 0.107  # TODO get this from URDF maybe?

        # Initial states
        self.inital_left_tcp = None
        self.inital_right_tcp = None
        self.initial_left_openness_state = None
        self.initial_right_openness_state = None

        self.left_tcp_trajectory = None
        self.right_tcp_trajectory = None

        self.left_openness_trajectory = None
        self.right_openness_trajectory = None

        # For visualization
        self.left_tcp_keyposes = None
        self.right_tcp_keyposes = None

        self.meshcat = meshcat

        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Plan)

    def Initialize(self, context, state):
        self.inital_left_tcp = self.GetInputPort("left_tcp").Eval(context)
        self.inital_right_tcp = self.GetInputPort("right_tcp").Eval(context)
        self.initial_left_openness_state = self.GetInputPort("left_openness_state").Eval(context)
        self.initial_right_openness_state = self.GetInputPort("right_openness_state").Eval(context)

        left_gripper_openness = self.initial_left_openness_state[0]
        right_gripper_openness = self.initial_right_openness_state[0]

        left_tcp_keyposes = {"initial": self.inital_left_tcp, "hold": self.inital_left_tcp}
        right_tcp_keyposes = {"initial": self.inital_right_tcp, "hold": self.inital_right_tcp}

        # Simply hold the initialization.
        self.left_tcp_trajectory = PiecewisePose.MakeLinear([0.0, 1.0], list(left_tcp_keyposes.values()))
        self.left_tcp_keyposes = left_tcp_keyposes
        self.right_tcp_trajectory = PiecewisePose.MakeLinear([0.0, 1.0], list(right_tcp_keyposes.values()))
        self.right_tcp_keyposes = right_tcp_keyposes

        self.left_openness_trajectory = PiecewisePolynomial.ZeroOrderHold(
            [0.0, 1.0], [[left_gripper_openness, left_gripper_openness]]
        )
        self.right_openness_trajectory = PiecewisePolynomial.ZeroOrderHold(
            [0.0, 1.0], [[right_gripper_openness, right_gripper_openness]]
        )

    def Plan(self, context, state):
        """Sets the trajectories for the right and left gripper poses and states."""
        # Implement you planning here, e.g. update the trajectories.
        self.UpdatePlanVisualization()

    def UpdatePlanVisualization(self):
        VisualizePoseTrajectory(self.meshcat, "left_tcp_trajectory", self.left_tcp_trajectory, self.left_tcp_keyposes)
        VisualizePoseTrajectory(
            self.meshcat, "right_tcp_trajectory", self.right_tcp_trajectory, self.right_tcp_keyposes
        )

    def OutputLeftTCP(self, context, output):
        left_tcp_pose = self.left_tcp_trajectory.GetPose(context.get_time())
        output.set_value(left_tcp_pose)

    def OutputRightTCP(self, context, output):
        right_tcp_pose = self.right_tcp_trajectory.GetPose(context.get_time())
        output.set_value(right_tcp_pose)

    def OutputLeftOpenness(self, context, output):
        output.SetFromVector(self.left_openness_trajectory.value(context.get_time()))

    def OutputRightOpenness(self, context, output):
        output.SetFromVector(self.right_openness_trajectory.value(context.get_time()))

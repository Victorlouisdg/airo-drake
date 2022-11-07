import numpy as np
from pydrake.geometry import Cylinder, Rgba
from pydrake.math import RigidTransform, RotationMatrix


def AddMeshcatTriad(meshcat, path, length=0.05, radius=0.002, opacity=1.0, X_PT=RigidTransform()):
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0])
    meshcat.SetTransform(path + "/x-axis", X_TG)
    meshcat.SetObject(path + "/x-axis", Cylinder(radius, length), Rgba(1, 0, 0, opacity))

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0])
    meshcat.SetTransform(path + "/y-axis", X_TG)
    meshcat.SetObject(path + "/y-axis", Cylinder(radius, length), Rgba(0, 1, 0, opacity))

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    meshcat.SetTransform(path + "/z-axis", X_TG)
    meshcat.SetObject(path + "/z-axis", Cylinder(radius, length), Rgba(0, 0, 1, opacity))


def VisualizePoseTrajectory(meshcat, path, trajectory, key_poses={}):
        traj_X_G = trajectory
        traj_p_G = traj_X_G.get_position_trajectory()
        p_G = traj_p_G.vector_values(traj_p_G.get_segment_times())

        starts = p_G[:, :-1]
        ends = p_G[:, 1:]
        meshcat.SetLineSegments(path, starts, ends, 2.0, rgba=Rgba(1, 0.65, 0))

        for name, X in key_poses.items():
            AddMeshcatTriad(meshcat, f"X_G{name}", X_PT=X)

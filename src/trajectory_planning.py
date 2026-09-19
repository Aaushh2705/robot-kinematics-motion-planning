import os
import numpy as np
import matplotlib.pyplot as plt

from inverse_kinematics import (
    forward_kinematics,
    inverse_kinematics
)


# --------------------------------------------------
# Cubic time scaling
# --------------------------------------------------

def cubic_time_scaling(t, total_time):
    """
    Cubic time-scaling function.

    s(0) = 0
    s(T) = 1
    s_dot(0) = 0
    s_dot(T) = 0
    """

    tau = t / total_time

    s = 3.0 * tau**2 - 2.0 * tau**3

    s_dot = (
        6.0 * tau * (1.0 - tau)
        / total_time
    )

    return s, s_dot


# --------------------------------------------------
# Joint-space trajectory generation
# --------------------------------------------------

def generate_joint_trajectory(
    q_start,
    q_goal,
    total_time=5.0,
    num_points=101
):
    time = np.linspace(
        0.0,
        total_time,
        num_points
    )

    q_trajectory = []
    q_velocity = []

    delta_q = q_goal - q_start

    for t in time:

        s, s_dot = cubic_time_scaling(
            t,
            total_time
        )

        q = q_start + s * delta_q
        q_dot = s_dot * delta_q

        q_trajectory.append(q)
        q_velocity.append(q_dot)

    return (
        time,
        np.array(q_trajectory),
        np.array(q_velocity)
    )


# --------------------------------------------------
# Cartesian end-effector trajectory
# --------------------------------------------------

def calculate_end_effector_path(q_trajectory):

    positions = []

    for q in q_trajectory:

        position = forward_kinematics(q)

        positions.append(position)

    return np.array(positions)


# --------------------------------------------------
# Plot joint trajectory
# --------------------------------------------------

def plot_joint_trajectory(
    time,
    q_trajectory
):

    q_degrees = np.rad2deg(
        q_trajectory
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        time,
        q_degrees[:, 0],
        linewidth=2,
        label="q1"
    )

    plt.plot(
        time,
        q_degrees[:, 1],
        linewidth=2,
        label="q2"
    )

    plt.plot(
        time,
        q_degrees[:, 2],
        linewidth=2,
        label="q3"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Joint Angle [deg]")

    plt.title(
        "3-DOF Planar Robot - Joint-Space Trajectory"
    )

    plt.grid(True)
    plt.legend()

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = (
        "results/joint_trajectory.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Joint trajectory saved to:",
        output_path
    )


# --------------------------------------------------
# Plot end-effector trajectory
# --------------------------------------------------

def plot_end_effector_trajectory(
    positions,
    target
):

    plt.figure(figsize=(8, 8))

    # Actual Cartesian positions generated
    # from FK of the joint trajectory
    plt.plot(
        positions[:, 0],
        positions[:, 1],
        "o-",
        markevery=10,
        linewidth=2,
        markersize=5,
        label="End-Effector Path"
    )

    # Start position
    plt.scatter(
        positions[0, 0],
        positions[0, 1],
        s=100,
        label="Start"
    )

    # Final position
    plt.scatter(
        positions[-1, 0],
        positions[-1, 1],
        s=100,
        label="Final Position"
    )

    # Desired Cartesian target
    plt.scatter(
        target[0],
        target[1],
        marker="x",
        s=160,
        linewidths=3,
        label="Target"
    )

    plt.xlabel("X Position [m]")
    plt.ylabel("Y Position [m]")

    plt.title(
        "3-DOF Planar Robot - End-Effector Trajectory"
    )

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    output_path = (
        "results/end_effector_trajectory.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "End-effector trajectory saved to:",
        output_path
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    # ----------------------------------------------
    # Initial robot configuration
    # ----------------------------------------------

    q_start = np.deg2rad([
        10.0,
        20.0,
        10.0
    ])

    # Cartesian goal
    target = np.array([
        0.40,
        0.45
    ])

    # ----------------------------------------------
    # Solve IK to obtain goal joint configuration
    # ----------------------------------------------

    q_goal, _, _ = inverse_kinematics(
        target,
        q_start
    )

    print()
    print("======================================")
    print(" TRAJECTORY PLANNING")
    print("======================================")

    print(
        "Start configuration [deg]:",
        np.rad2deg(q_start)
    )

    print(
        "Goal configuration [deg]:",
        np.rad2deg(q_goal)
    )

    # ----------------------------------------------
    # Generate smooth joint trajectory
    # ----------------------------------------------

    total_time = 5.0
    num_points = 101

    (
        time,
        q_trajectory,
        q_velocity
    ) = generate_joint_trajectory(
        q_start,
        q_goal,
        total_time,
        num_points
    )

    # ----------------------------------------------
    # Calculate REAL end-effector positions using FK
    # ----------------------------------------------

    positions = calculate_end_effector_path(
        q_trajectory
    )

    final_position = positions[-1]

    final_error = np.linalg.norm(
        target - final_position
    )

    # ----------------------------------------------
    # Verification
    # ----------------------------------------------

    print()
    print(
        "Trajectory duration:",
        total_time,
        "s"
    )

    print(
        "Trajectory samples:",
        num_points
    )

    print()
    print(
        "Initial joint velocity [deg/s]:",
        np.rad2deg(q_velocity[0])
    )

    print(
        "Final joint velocity [deg/s]:",
        np.rad2deg(q_velocity[-1])
    )

    print()
    print(
        "Initial end-effector position:",
        positions[0]
    )

    print(
        "Final end-effector position:",
        final_position
    )

    print(
        "Target:",
        target
    )

    print(
        "Final Cartesian error:",
        final_error,
        "m"
    )

    print("======================================")
    print()

    # ----------------------------------------------
    # Generate figures
    # ----------------------------------------------

    plot_joint_trajectory(
        time,
        q_trajectory
    )

    plot_end_effector_trajectory(
        positions,
        target
    )

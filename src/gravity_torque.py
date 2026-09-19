import os
import numpy as np
import matplotlib.pyplot as plt

from inverse_kinematics import inverse_kinematics
from trajectory_planning import generate_joint_trajectory


# --------------------------------------------------
# Robot geometry
# --------------------------------------------------

L1 = 0.25
L2 = 0.30
L3 = 0.20

# Centre-of-mass locations measured from
# the beginning of each link.
LC1 = L1 / 2.0
LC2 = L2 / 2.0
LC3 = L3 / 2.0


# --------------------------------------------------
# Nominal dynamic model parameters
# --------------------------------------------------
#
# These are MODEL ASSUMPTIONS, not measured hardware
# parameters. Replace them with measured values if a
# physical robot is used later.
#

M1 = 1.5
M2 = 1.0
M3 = 0.5

G = 9.81


# --------------------------------------------------
# Gravity compensation torque
# --------------------------------------------------

def gravity_torque(q):
    """
    Calculate gravity-compensation joint torques
    for a 3-DOF planar manipulator operating in
    the vertical x-y plane.

    Gravity acts in the negative y direction.

    Parameters
    ----------
    q : array-like
        Joint angles [q1, q2, q3] in radians.

    Returns
    -------
    numpy.ndarray
        Gravity compensation torques
        [tau1, tau2, tau3] in N*m.
    """

    q1, q2, q3 = q

    q12 = q1 + q2
    q123 = q1 + q2 + q3

    tau1 = G * (
        M1 * LC1 * np.cos(q1)
        + M2 * (
            L1 * np.cos(q1)
            + LC2 * np.cos(q12)
        )
        + M3 * (
            L1 * np.cos(q1)
            + L2 * np.cos(q12)
            + LC3 * np.cos(q123)
        )
    )

    tau2 = G * (
        M2 * LC2 * np.cos(q12)
        + M3 * (
            L2 * np.cos(q12)
            + LC3 * np.cos(q123)
        )
    )

    tau3 = G * (
        M3 * LC3 * np.cos(q123)
    )

    return np.array([
        tau1,
        tau2,
        tau3
    ])


# --------------------------------------------------
# Potential energy
# --------------------------------------------------

def potential_energy(q):
    """
    Calculate gravitational potential energy.

    This function is also used to independently
    verify the analytical gravity torque equations.
    """

    q1, q2, q3 = q

    q12 = q1 + q2
    q123 = q1 + q2 + q3

    # Vertical position of each link COM

    y1 = (
        LC1 * np.sin(q1)
    )

    y2 = (
        L1 * np.sin(q1)
        + LC2 * np.sin(q12)
    )

    y3 = (
        L1 * np.sin(q1)
        + L2 * np.sin(q12)
        + LC3 * np.sin(q123)
    )

    U = G * (
        M1 * y1
        + M2 * y2
        + M3 * y3
    )

    return U


# --------------------------------------------------
# Numerical verification
# --------------------------------------------------

def numerical_gravity_torque(q, epsilon=1e-6):
    """
    Numerically evaluate dU/dq using central
    finite differences.
    """

    tau = np.zeros(3)

    for i in range(3):

        dq = np.zeros(3)
        dq[i] = epsilon

        U_plus = potential_energy(q + dq)
        U_minus = potential_energy(q - dq)

        tau[i] = (
            U_plus - U_minus
        ) / (2.0 * epsilon)

    return tau


# --------------------------------------------------
# Calculate torque along trajectory
# --------------------------------------------------

def calculate_torque_trajectory(q_trajectory):

    torque_trajectory = []

    for q in q_trajectory:

        tau = gravity_torque(q)

        torque_trajectory.append(tau)

    return np.array(torque_trajectory)


# --------------------------------------------------
# Plot gravity torque
# --------------------------------------------------

def plot_gravity_torque(
    time,
    torque_trajectory
):

    plt.figure(figsize=(9, 6))

    plt.plot(
        time,
        torque_trajectory[:, 0],
        linewidth=2,
        label="tau1"
    )

    plt.plot(
        time,
        torque_trajectory[:, 1],
        linewidth=2,
        label="tau2"
    )

    plt.plot(
        time,
        torque_trajectory[:, 2],
        linewidth=2,
        label="tau3"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Gravity Compensation Torque [N m]")

    plt.title(
        "3-DOF Planar Robot - Gravity Compensation Torque"
    )

    plt.grid(True)
    plt.legend()

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = (
        "results/gravity_torque.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Gravity torque plot saved to:",
        output_path
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    # Initial configuration
    q_start = np.deg2rad([
        10.0,
        20.0,
        10.0
    ])

    # Same Cartesian target used by the IK
    # and trajectory-planning modules
    target = np.array([
        0.40,
        0.45
    ])

    # --------------------------------------------------
    # Obtain actual IK goal configuration
    # --------------------------------------------------

    q_goal, _, _ = inverse_kinematics(
        target,
        q_start
    )

    # --------------------------------------------------
    # Generate the same trajectory used previously
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Gravity torque along trajectory
    # --------------------------------------------------

    torque_trajectory = (
        calculate_torque_trajectory(
            q_trajectory
        )
    )

    # --------------------------------------------------
    # Verify analytical torque model
    # --------------------------------------------------

    analytical_tau = gravity_torque(
        q_goal
    )

    numerical_tau = numerical_gravity_torque(
        q_goal
    )

    verification_error = np.max(
        np.abs(
            analytical_tau
            - numerical_tau
        )
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("======================================")
    print(" GRAVITY TORQUE ANALYSIS")
    print("======================================")

    print()
    print("Model assumptions:")
    print("m1 =", M1, "kg")
    print("m2 =", M2, "kg")
    print("m3 =", M3, "kg")
    print("g  =", G, "m/s^2")

    print()
    print(
        "Start configuration [deg]:",
        np.rad2deg(q_start)
    )

    print(
        "Goal configuration [deg]:",
        np.rad2deg(q_goal)
    )

    print()
    print(
        "Initial gravity torque [N m]:",
        torque_trajectory[0]
    )

    print(
        "Final gravity torque [N m]:",
        torque_trajectory[-1]
    )

    print()
    print(
        "Maximum absolute torque [N m]:",
        np.max(
            np.abs(torque_trajectory),
            axis=0
        )
    )

    print()
    print("--------------------------------------")
    print(" MODEL VERIFICATION")
    print("--------------------------------------")

    print(
        "Analytical torque [N m]:",
        analytical_tau
    )

    print(
        "Numerical dU/dq [N m]:",
        numerical_tau
    )

    print(
        "Maximum verification error:",
        verification_error
    )

    print("======================================")
    print()

    # --------------------------------------------------
    # Generate plot
    # --------------------------------------------------

    plot_gravity_torque(
        time,
        torque_trajectory
    )

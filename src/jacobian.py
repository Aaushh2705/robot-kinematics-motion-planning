import numpy as np

# Link lengths [m]
L1 = 0.25
L2 = 0.30
L3 = 0.20


def forward_position(q):
    """
    End-effector position [x, y]
    for a 3-DOF planar serial manipulator.
    """
    q1, q2, q3 = q

    x = (
        L1 * np.cos(q1)
        + L2 * np.cos(q1 + q2)
        + L3 * np.cos(q1 + q2 + q3)
    )

    y = (
        L1 * np.sin(q1)
        + L2 * np.sin(q1 + q2)
        + L3 * np.sin(q1 + q2 + q3)
    )

    return np.array([x, y])


def analytical_jacobian(q):
    """
    Analytical 2x3 Jacobian:
    maps joint velocities to end-effector planar velocity.
    """
    q1, q2, q3 = q

    s1 = np.sin(q1)
    s12 = np.sin(q1 + q2)
    s123 = np.sin(q1 + q2 + q3)

    c1 = np.cos(q1)
    c12 = np.cos(q1 + q2)
    c123 = np.cos(q1 + q2 + q3)

    dx_dq1 = -L1 * s1 - L2 * s12 - L3 * s123
    dx_dq2 = -L2 * s12 - L3 * s123
    dx_dq3 = -L3 * s123

    dy_dq1 = L1 * c1 + L2 * c12 + L3 * c123
    dy_dq2 = L2 * c12 + L3 * c123
    dy_dq3 = L3 * c123

    return np.array([
        [dx_dq1, dx_dq2, dx_dq3],
        [dy_dq1, dy_dq2, dy_dq3]
    ])


def numerical_jacobian(q, eps=1e-6):
    """
    Numerical Jacobian using finite differences.
    Used to validate the analytical expression.
    """
    J = np.zeros((2, 3))
    p0 = forward_position(q)

    for i in range(3):
        q_perturbed = q.copy()
        q_perturbed[i] += eps

        p1 = forward_position(q_perturbed)

        J[:, i] = (p1 - p0) / eps

    return J


if __name__ == "__main__":
    q_deg = np.array([30.0, 45.0, -20.0])
    q = np.deg2rad(q_deg)

    J_analytical = analytical_jacobian(q)
    J_numerical = numerical_jacobian(q)

    error = J_analytical - J_numerical

    print("Joint angles [deg]:")
    print(q_deg)

    print("\nAnalytical Jacobian:")
    print(J_analytical)

    print("\nNumerical Jacobian:")
    print(J_numerical)

    print("\nDifference:")
    print(error)

    print("\nMaximum absolute error:")
    print(np.max(np.abs(error)))

import numpy as np

# Link lengths in meters
L1 = 0.25
L2 = 0.30
L3 = 0.20

# Joint angles in degrees
q1_deg = 30
q2_deg = 45
q3_deg = -20

# Convert to radians
q1 = np.deg2rad(q1_deg)
q2 = np.deg2rad(q2_deg)
q3 = np.deg2rad(q3_deg)

def transform_z(theta, length):
    """
    Homogeneous transformation:
    rotation about Z followed by translation along X.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, length * np.cos(theta)],
        [np.sin(theta),  np.cos(theta), 0, length * np.sin(theta)],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])

T01 = transform_z(q1, L1)
T12 = transform_z(q2, L2)
T23 = transform_z(q3, L3)

T03 = T01 @ T12 @ T23

end_effector = T03[:3, 3]

print("Final transformation matrix:")
print(T03)

print("\nEnd-effector position [x, y, z]:")
print(end_effector)

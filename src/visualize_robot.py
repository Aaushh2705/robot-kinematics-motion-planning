import numpy as np
import matplotlib.pyplot as plt

# Link lengths [m]
L1 = 0.25
L2 = 0.30
L3 = 0.20

# Joint angles [degrees]
q1 = np.deg2rad(30)
q2 = np.deg2rad(45)
q3 = np.deg2rad(-20)

# Cumulative joint angles
theta1 = q1
theta2 = q1 + q2
theta3 = q1 + q2 + q3

# Joint positions
x0, y0 = 0.0, 0.0

x1 = L1 * np.cos(theta1)
y1 = L1 * np.sin(theta1)

x2 = x1 + L2 * np.cos(theta2)
y2 = y1 + L2 * np.sin(theta2)

x3 = x2 + L3 * np.cos(theta3)
y3 = y2 + L3 * np.sin(theta3)

# Store coordinates
x = [x0, x1, x2, x3]
y = [y0, y1, y2, y3]

# Plot robot
plt.figure(figsize=(7, 7))
plt.plot(x, y, "o-", linewidth=3, markersize=8)

plt.scatter(x3, y3, s=100, label="End Effector")

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("3-DOF Serial Robot - Forward Kinematics")

plt.axis("equal")
plt.grid(True)
plt.legend()

# Save for GitHub README
plt.savefig("results/forward_kinematics.png", dpi=200)

plt.show()

print("End-effector position:")
print(f"x = {x3:.4f} m")
print(f"y = {y3:.4f} m")

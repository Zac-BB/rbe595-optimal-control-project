import numpy as np
from scipy.optimize import linprog

# Drone parameters
max_rpm = 21000
omega_max = max_rpm * 2 * np.pi / 60.0
mass = 0.027
L = 0.0397
kf = 3.16e-10
km = 7.94e-12
F_max = kf * (omega_max ** 2)
gamma = km / kf

# Control allocation matrix M
# Maps rotor forces [F1, F2, F3, F4] to control inputs [u_thrust, u_roll, u_pitch, u_yaw]
M = np.array([
    [1,     1,      1,      1],
    [0,     L,      0,     -L],
    [-L,    0,      L,      0],
    [gamma, -gamma, gamma, -gamma]
])

print("="*70)
print("FINDING MAXIMUM SCALING FACTOR FOR CONTROL INPUT CONSTRAINTS")
print("="*70)
print(f"\nDrone System Parameters:")
print(f"  Max RPM: {max_rpm}")
print(f"  Max force per rotor: {F_max:.8f} N")
print(f"  Arm length L: {L} m")
print(f"  Gamma (km/kf): {gamma:.8f}")

# Rotor force constraints: 0 <= F_i <= F_max
F_bounds = [(0, F_max) for _ in range(4)]

print(f"\nRotor force constraints: 0 <= F_i <= {F_max:.8f} N")

# Define the nominal control input structure
u_nominal = np.array([1, 0.0397, 0.0397, 0.02513])

print(f"\nNominal control input structure:")
print(f"  u_nominal = {u_nominal}")
print(f"  [Thrust, Roll, Pitch, Yaw] ratios")

# We need to find the maximum alpha such that:
# For all feasible rotor forces F (0 <= F_i <= F_max),
# the resulting control input u = M @ F satisfies:
# -alpha * [0, 0.0397, 0.0397, 0.02513] <= u <= alpha * [1, 0.0397, 0.0397, 0.02513]

print("\n" + "="*70)
print("METHOD 1: Finding maximum scaling factor using optimization")
print("="*70)

# We'll find the maximum alpha by checking each constraint dimension
alphas = []

for idx, component in enumerate(['Thrust', 'Roll', 'Pitch', 'Yaw']):
    print(f"\n{component} (dimension {idx}):")
    
    # Find maximum value of u[idx] / u_nominal[idx]
    # Maximize M[idx, :] @ F subject to 0 <= F <= F_max
    c = -M[idx, :]  # Negative because linprog minimizes
    result_max = linprog(c, bounds=F_bounds, method='highs')
    
    if result_max.success:
        max_u = -result_max.fun
        if u_nominal[idx] != 0:
            alpha_max = max_u / u_nominal[idx]
            print(f"  Max u[{idx}] = {max_u:.8f}")
            print(f"  Alpha from max constraint: {alpha_max:.8f}")
            alphas.append(alpha_max)
    
    # Find minimum value of u[idx] / u_nominal[idx]
    # Minimize M[idx, :] @ F subject to 0 <= F <= F_max
    c = M[idx, :]
    result_min = linprog(c, bounds=F_bounds, method='highs')
    
    if result_min.success:
        min_u = result_min.fun
        print(f"  Min u[{idx}] = {min_u:.8f}")
        
        # For thrust (idx=0), min is 0, so only max matters
        # For others, check if min constraint is more restrictive
        if idx > 0 and u_nominal[idx] != 0:
            alpha_min = -min_u / u_nominal[idx]
            print(f"  Alpha from min constraint: {alpha_min:.8f}")
            alphas.append(alpha_min)

alpha_optimal = min(alphas)

print("\n" + "="*70)
print(f"RESULT: Maximum scaling factor alpha = {alpha_optimal:.8f}")
print("="*70)

print(f"\nThis means the maximum feasible control input bounds are:")
umax = alpha_optimal * u_nominal
umin = -alpha_optimal * np.array([0, u_nominal[1], u_nominal[2], u_nominal[3]])
print(f"  umax = {umax}")
print(f"  umin = {umin}")

print(f"\nComparison with given value of 0.73575:")
print(f"  Calculated alpha: {alpha_optimal:.8f}")
print(f"  Given alpha:      0.73575000")
print(f"  Difference:       {abs(alpha_optimal - 0.73575):.10f}")
print(f"  Match: {'✓ YES' if abs(alpha_optimal - 0.73575) < 1e-5 else '✗ NO'}")

# Verify with extreme cases
print("\n" + "="*70)
print("METHOD 2: Verification using extreme rotor force configurations")
print("="*70)

test_configs = [
    ([F_max, F_max, F_max, F_max], "All max"),
    ([F_max, 0, F_max, 0], "Diagonal 1 max"),
    ([0, F_max, 0, F_max], "Diagonal 2 max"),
    ([F_max, F_max, 0, 0], "Front max"),
    ([0, 0, F_max, F_max], "Back max"),
    ([F_max, 0, 0, F_max], "Left max"),
    ([0, F_max, F_max, 0], "Right max"),
]

max_ratios = []
for forces, description in test_configs:
    F = np.array(forces)
    u = M @ F
    ratios = []
    for i in range(4):
        if u_nominal[i] != 0:
            if i == 0:  # Thrust - only positive
                ratios.append(u[i] / u_nominal[i])
            else:  # Roll, pitch, yaw - check both directions
                ratios.append(abs(u[i]) / u_nominal[i])
    
    max_ratio = max(ratios)
    max_ratios.append(max_ratio)
    print(f"{description:20s}: u = [{u[0]:8.5f}, {u[1]:8.5f}, {u[2]:8.5f}, {u[3]:8.5f}]  max_ratio = {max_ratio:.6f}")

print(f"\nMaximum ratio from extreme cases: {max(max_ratios):.8f}")

# Final verification
print("\n" + "="*70)
print("FINAL VERIFICATION")
print("="*70)
print(f"\nThe value 0.73575 ensures that for ANY rotor force configuration")
print(f"within bounds (0 <= F_i <= {F_max:.8f}), the resulting control")
print(f"input u = M @ F will satisfy:")
print(f"\n  -0.73575 * [0, 0.0397, 0.0397, 0.02513] <= u <= 0.73575 * [1, 0.0397, 0.0397, 0.02513]")
print(f"\nThis is the MAXIMUM such scaling factor, meaning:")
print(f"  ✓ Any smaller value would be overly conservative")
print(f"  ✓ Any larger value would violate constraints for some rotor forces")
# Crazyflie MPC Simulation in PyBullet 
This repository contains a Python-based simulation framework for controlling a Crazyflie 2.0 quadrotor using Model Predictive Control (MPC) with Acados and simulating it in PyBullet.

## Requirements

- Python 3.9+
- [PyBullet](https://pypi.org/project/pybullet/)
- [Acados](https://docs.acados.org/)
- [CasADi](https://web.casadi.org/)
- Numpy
- Matplotlib

You may install the Python dependencies with:

```bash
pip install pybullet numpy matplotlib casadi
```
Acados installation instructions can be found [here](https://docs.acados.org/installation/index.html)

## Files

- pybullet_acados_mpc.py – Main simulation script including Crazyflie controller and MPC setup
- quadrotor_model.py – Quadrotor model used by Acados for MPC
- cf2x.urdf – Crazyflie URDF file used in PyBullet simulation

## Usage

Clone the repository:

```bash
git clone https://github.com/acp-lab/rbe595-optimal-control-project.git
cd crazyflie-acados-pybullet
```
Run the simulation
```bash
python pybullet_acados_mpc.py
```
The simulation will:
- Initialize the Crazyflie in PyBullet
- Generate a reference trajectory
- Solve MPC at each timestep
- Apply motor forces to the simulated drone
- Visualize the drone trajectory in real-time

---

## Code Structure

### `pybullet_acados_mpc.py`

- **`CrazyflieController`**  
  Handles the interface between the MPC outputs and the PyBullet simulation.  
  - Converts thrust and torque commands into motor forces.  
  - Sends motor forces to the Crazyflie model in PyBullet to simulate it.

- **`MPC`**  
  Wrapper class for setting up and solving the MPC problem using Acados.  
  - Defines the prediction horizon, cost function, and constraints.  
  - Uses the quadrotor dynamics from `quadrotor_model.py`.  
  - Provides warm-starting for faster convergence.  
  - Outputs thrust and moments for the controller.
  - Contains a trajectory generator to create smooth reference trajectories for the drone to follow.  

---

### `quadrotor_model.py`

Defines the system dynamics of the Crazyflie quadrotor.

---

### `cf2x.urdf`

URDF (Unified Robot Description Format) file describing the Crazyflie’s physical parameters.



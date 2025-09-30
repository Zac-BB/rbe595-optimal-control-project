# Check f_impl, meta information, x_dot equations
import numpy as np
from acados_template import AcadosModel
from casadi import SX, MX, vertcat, cos, sin, sqrt, horzcat, transpose, inv, Function

def quadrotor_model_auto() -> AcadosModel:
    """
    Create a quadrotor model for acados MPC
    
    State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
    - Position: (x, y, z)
    - Velocity: (vx, vy, vz) 
    - Attitude: (qw, qx, qy, qz)
    - Angular rates: (p, q, r)
    
    Control vector: [f, tau_phi, tau_theta, tau_psi]
    - f: total thrust
    - tau_phi, tau_theta, tau_psi: torques around body axes
    """
    
    model_name = 'quadrotor'
    
    
    return model

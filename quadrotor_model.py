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
    
    g = 9.81
    Ixx = 1.4
    Iyy = 1.4
    Izz = 2.17
    l = 0.046
    m = 0.027
    mu = 0.73575


    x = SX.sym("x")
    y = SX.sym("y")
    z = SX.sym("z")
    vx = SX.sym("vx")
    vy = SX.sym("vy")
    vz = SX.sym("vz")
    qw = SX.sym("qw")
    qx = SX.sym("qx")
    qy = SX.sym("qy")
    qz = SX.sym("qz")
    p = SX.sym("p")
    q = SX.sym("q")
    r = SX.sym("r")
    X = vertcat(x,y,z,vx,vy,vz,qw,qx,qy,qz,p,q,r)
    X_dot = SX.sym('xdot', 13)
    T = SX.sym('u_1')
    tau_x = SX.sym('tau_x')
    tau_y = SX.sym('tau_y')
    tau_z = SX.sym('tau_z')
    u = vertcat(T,tau_x,tau_y,tau_z)
    R_AB = vertcat(
        horzcat(1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)),
        horzcat(2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)),
        horzcat(2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2))
    )
    
    
    Omega = vertcat(
        horzcat(0, -p, -q, -r),
        horzcat(p, 0, r, -q),
        horzcat(q, -r, 0, p),
        horzcat(r, q, -p, 0)
    )
    v_d = SX([[0],[0],[-g]]) + (1/m)*R_AB@vertcat(0,0,T)
    q_d = 0.5*Omega@vertcat(qw,qx,qy,qz)
    I = SX([[Ixx,0,0],[0,Iyy,0],[0,0,Izz]])
    I_inv = SX([[1/Ixx,0,0],[0,1/Iyy,0],[0,0,1/Izz]])
    Sk_omega = vertcat(horzcat(0,-r,q),horzcat(r,0,p),horzcat(-q,p,0))

    omega_d = I_inv@vertcat(tau_x,tau_y,tau_z) - I_inv@Sk_omega@I@vertcat(p,q,r)
    f_expl = vertcat(vx,vy,vz,v_d,q_d,omega_d)
    model = AcadosModel()

    model.f_expl_expr = f_expl
    model.f_impl_expr = X_dot - f_expl
    model.x = X
    model.u = u
    model.p = []
    model.name = model_name

    return model

if __name__ == "__main__":
    quadrotor_model_auto()
import pybullet as p
import pybullet_data
import numpy as np
import time
import math

from acados_template import AcadosOcp, AcadosOcpSolver, plot_trajectories
from quadrotor_model import quadrotor_model_auto
import casadi as ca
import matplotlib.pyplot as plt

class CrazyflieController:
    def __init__(self, urdf_path="./cf2x.urdf"):
        # Physical parameters from URDF
        self.mass = 0.027 # 0.027  # kg
        self.arm_length = 0.0397  # m
        self.kf = 3.16e-10  # thrust coefficient
        self.km = 7.94e-12  # moment coefficient
        self.max_rpm = 21000  # typical max RPM for CF2.0
        omega_max = self.max_rpm * 2*np.pi / 60.0  # rad/s
        self.max_thrust_per_motor = self.kf * (omega_max ** 2)
        
        # Initialize PyBullet
        self.setup_pybullet(urdf_path)
    
    def quaternion2rotation_matrix(self, quaternion):
        """
        quaternion: [x,y,z,w] (shape used in Bullet Physics) to rotation matrix 
        """
        n = np.dot(quaternion,quaternion)
        if n<1e-12:
            return np.identity(3)
        q =np.array([quaternion[3],quaternion[0],quaternion[1],quaternion[2]])
        q *= math.sqrt(2.0/n)
        q = np.outer(q,q)
        return np.array([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])

    def setup_pybullet(self, urdf_path):
        """Initialize PyBullet simulation"""
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        dt = 1/10.0
        p.setTimeStep(dt)
        p.setPhysicsEngineParameter(numSubSteps=10)
        # Load the Crazyflie
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        plane = p.loadURDF("plane.urdf")
        self.cf_id = p.loadURDF(urdf_path, start_pos, start_orientation, flags=p.URDF_USE_INERTIA_FROM_FILE)
        
        # Get motor link indices (prop links)
        self.motor_links = []
        for i in range(4):
            link_name = f"prop{i}_link"
            for j in range(p.getNumJoints(self.cf_id)):
                link_info = p.getJointInfo(self.cf_id, j)
                if link_info[12].decode('utf-8') == link_name:
                    self.motor_links.append(j)
                    break
        
        # p.changeDynamics(bodyUniqueId=self.cf_id, linkIndex=-1, lateralFriction=0.0)
        print(f"PyBullet Dynamics: {p.getDynamicsInfo(self.cf_id, -1)}")
        print(f"Found motor links at indices: {self.motor_links}")
        
    def mpc_to_motor_forces(self, thrust, moment_x, moment_y, moment_z):
        """
        Convert MPC outputs (thrust, moments) to individual motor forces
        
        Args:
            thrust: Total thrust command [N]
            moment_x: Roll moment [N⋅m]
            moment_y: Pitch moment [N⋅m] 
            moment_z: Yaw moment [N⋅m]
            
        Returns:
            motor_forces: Array of 4 motor forces [N]
        """
        c = self.km / self.kf
        control_vector = np.array([thrust, moment_x, moment_y, moment_z])

        mat = np.array([[1, 1, 1, 1], 
                        [self.arm_length/math.sqrt(2), -self.arm_length/math.sqrt(2), self.arm_length/math.sqrt(2), -self.arm_length/math.sqrt(2)], 
                        [-self.arm_length/math.sqrt(2), -self.arm_length/math.sqrt(2), self.arm_length/math.sqrt(2), self.arm_length/math.sqrt(2)],
                        [c, -c, -c, c]])

        # [thrust, roll_moment, pitch_moment, yaw_moment] = mat * [F0, F1, F2, F3]
        motor_forces = np.linalg.inv(mat) @ control_vector

        # print("Control vector: ", control_vector)
        
        return motor_forces

    def apply_motor_forces(self, motor_forces, cur_traj):
        """
        Apply thrust forces at the four rotors.
        """
        pos, orn = p.getBasePositionAndOrientation(self.cf_id)
        # print(cur_traj[6:10])
        quat = cur_traj[6:10]
        des_quat = [quat[1], quat[2], quat[3], quat[0]]
        # print(des_quat)
        quat_norm_sq = des_quat[0]*des_quat[0] + des_quat[1]*des_quat[1] + des_quat[2]*des_quat[2] + des_quat[3]*des_quat[3]
    
        quat_norm = math.sqrt(quat_norm_sq)
        des_quat[0] = des_quat[0] / quat_norm
        des_quat[1] = des_quat[1] / quat_norm
        des_quat[2] = des_quat[2] / quat_norm
        des_quat[3] = des_quat[3] / quat_norm
        # print("After normalizing", des_quat)
        R_meas = self.quaternion2rotation_matrix(des_quat)

        orn = np.array(orn)
        quat_norm_sq_orn = orn[0]*orn[0] + orn[1]*orn[1] + orn[2]*orn[2] + orn[3]*orn[3]
    
        quat_norm_orn = math.sqrt(quat_norm_sq_orn)
        orn[0] = orn[0] / quat_norm_orn
        orn[1] = orn[1] / quat_norm_orn
        orn[2] = orn[2] / quat_norm_orn
        orn[3] = orn[3] / quat_norm_orn
        # R_meas = self.quaternion2rotation_matrix(orn)

        # print("Motor Forces in body frame: ", motor_forces)
        motor_forces = np.array([f_i * R_meas[:,2] for f_i in motor_forces])
        # print("Forces applied at each motor: ",  motor_forces)

        p.applyExternalForce(self.cf_id,-1,motor_forces[0],[self.arm_length/math.sqrt(2), self.arm_length/math.sqrt(2),0.], p.LINK_FRAME)
        p.applyExternalForce(self.cf_id,-1,motor_forces[1],[self.arm_length/math.sqrt(2), -self.arm_length/math.sqrt(2),0.], p.LINK_FRAME)
        p.applyExternalForce(self.cf_id,-1,motor_forces[2],[-self.arm_length/math.sqrt(2), self.arm_length/math.sqrt(2),0.], p.LINK_FRAME)
        p.applyExternalForce(self.cf_id,-1,motor_forces[3],[-self.arm_length/math.sqrt(2), -self.arm_length/math.sqrt(2),0.], p.LINK_FRAME)

    def get_state(self):
        """Get current state of the Crazyflie for MPC feedback"""
        pos, orn = p.getBasePositionAndOrientation(self.cf_id)
        vel, ang_vel = p.getBaseVelocity(self.cf_id)
        euler = p.getEulerFromQuaternion(orn)
        
        state = {
            'position': np.array(pos),
            'velocity': np.array(vel),
            'orientation': np.array([orn[3], orn[0], orn[1], orn[2]]),  # [roll, pitch, yaw]            
            'angular_velocity': np.array(ang_vel)
        }
        return state
    
    def step_simulation(self, mpc_output, cur_traj):# dt=1/240):
        """
        One simulation step with MPC control
        
        Args:
            mpc_output: Dict with keys 'thrust', 'moment_x', 'moment_y', 'moment_z'
            dt: Time step
        """
        # Convert MPC output to motor forces
        motor_forces = self.mpc_to_motor_forces(
            mpc_output['thrust'],
            mpc_output['moment_x'], 
            mpc_output['moment_y'],
            mpc_output['moment_z']
        )
        
        # Apply forces
        self.apply_motor_forces(motor_forces, cur_traj)
        
        # Step simulation
        p.stepSimulation()
        # time.sleep(dt)
        
        return self.get_state()

class MPC:
    """ MPC for the drone"""
    def __init__(self,target_height):
        print("init")
        self.N = 20 
        self.dt = 0.1
        self.traj_time = 2.0

        self.mass = 0.027 # 0.027  # kg
        self.arm_length = 0.0397  # m
        self.kf = 3.16e-10  # thrust coefficient
        self.km = 7.94e-12  # moment coefficient
        self.max_rpm = 21000  # typical max RPM for CF2.0
        omega_max = self.max_rpm * 2*np.pi / 60.0  # rad/s
        self.max_thrust_per_motor = self.kf * (omega_max ** 2)

        self.target_height = target_height
        self.sim_steps = int(self.traj_time / self.dt)

        params = {
            't': [0, self.traj_time],
            'q': [[0.0, 0.0, 0.0], [0.0, 0.0, self.target_height]],
            'v': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            'a': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            'dt': self.dt
        }
        
        traj = self.make_trajectory('quintic', params)
        self.t = traj['t']
        n_points = len(traj['t'])
        self.trajectory1 = np.zeros((n_points, 13))

        self.trajectory1[:, 0:3] = traj['q']  # x, y, z
        self.trajectory1[:, 3:6] = traj['v']  # vx, vy, vz
        
        self.trajectory1[:, 6] = 1.0  # qw
        self.trajectory1[:, 7:10] = 0.0  # qx, qy, qz
        
        self.trajectory1[:, 10:13] = 0.0 # wx, wy, wz


        self.ocp = self.create_ocp()
        # self.solver = AcadosOcpSolver(self.ocp)
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=f'{self.ocp.model.name}_ocp.json', verbose=False)
        self.simX1 = np.zeros((self.sim_steps, 13))
        self.simU = np.zeros((self.sim_steps, 4))


    def solve(self, step, current_state):
        """
        Solve MPC at current step
        
        Args:
            step: Current simulation step
            current_state: Current state vector [pos(3), vel(3), quat(4), omega(3)]
            
        Returns:
            dict: Control commands with keys 'thrust', 'moment_x', 'moment_y', 'moment_z'
        """


        # self.ocp_solver.set(0, "x", current_state)
        self.ocp_solver.set(0, "lbx", current_state)
        self.ocp_solver.set(0, "ubx", current_state)


        # r_ref = np.array([0,0,self.target_height]).reshape(3,1)
        # v_ref = np.array([0,0,0]).reshape(3,1)
        # psi_ref = np.array([1,0,0,0]).reshape(4,1)
        # psi_dot_ref = np.array([0,0,0]).reshape(3,1) #forced 
        # 
        # for i in range(self.N):
        #     yref = np.vstack([r_ref,v_ref,psi_ref,psi_dot_ref,u0])
        #     self.ocp_solver.set(i, "yref", yref)


        for i in range(self.N):
            traj_idx = min(step + i, len(self.trajectory1) - 1)
            r_ref = self.trajectory1[traj_idx, 0:3].reshape(3,1)
            v_ref = self.trajectory1[traj_idx, 3:6].reshape(3,1)
            psi_ref = np.array([1,0,0,0]).reshape(4,1)
            psi_dot_ref = np.array([0,0,0]).reshape(3,1)
            u_0 = np.array([self.mass * 9.81, 0, 0, 0]).reshape(4,1)  
            yref = np.vstack([r_ref, v_ref, psi_ref, psi_dot_ref, u_0])
            self.ocp_solver.set(i, "yref", yref)

        yref_e = np.vstack([r_ref,psi_ref])
        self.ocp_solver.set(self.N, "yref", yref_e)
        # self.ocp_solver.set(self.N, "yref_e", yref_e)




        # self.ocp.cost.yref = np.vstack([r_ref,v_ref,psi_ref,psi_dot_ref,u0])
        # self.ocp.cost.yref_e = np.vstack([r_ref,psi_ref])
        # self.ocp.constraints.x0 = current_state






        status = self.ocp_solver.solve()

        if status != 0:
            print(f"[MPC] Solver failed with status: {status}")

        u = self.ocp_solver.get(0, "u")
        
        # u  = self.ocp_solver.solve_for_x0(current_state)
        # status = self.ocp_solver.get_status()
        # # status = self.ocp_solver.solve()
        # self.ocp_solver.print_statistics()
        # # if(status):
        # # u_opt = self.ocp_solver.get(0, "u")
        print("U: {}".format(u))
        


        return {
            'thrust':   u[0],
            'moment_x': u[1],
            'moment_y': u[2],
            'moment_z': u[3]
        }
    
    def create_ocp(self):         
        ocp = AcadosOcp()
        model = quadrotor_model_auto()
        ocp.model = model
        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx+nu
    
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.N*self.dt

        Qr = np.diag([10, 10, 10])
        Qv = np.diag([5,5,5])
        Qa = np.diag([0.5,0.5,0.5,0.5])
        Qw = np.diag([0.1,0.1,0.1])
        
        R = np.diag([0.0001, 0.01, 0.01, 0.01])  

        ocp.cost.cost_type = "LINEAR_LS"


        W = np.block([
            [Qr, np.zeros((3,14))],
            [np.zeros((3,3)), Qv,np.zeros((3,11))],
            [np.zeros((4,6)), Qa,np.zeros((4,7))],
            [np.zeros((3,10)), Qw,np.zeros((3,4))],
            [np.zeros((4,13)),R],
        ])

        ocp.cost.Vx = np.vstack([np.eye(nx),np.zeros((nu,nx))])
        ocp.cost.Vu = np.vstack([np.zeros((nx,nu)),np.eye(nu)])
        ocp.cost.W = W
        ocp.cost.yref = np.zeros((17,1))
 

        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.Vx_e = np.block([[np.eye(3),np.zeros((3,10))],
                                  [np.zeros((4,6)),np.eye(4),np.zeros((4,3))]])
        ocp.cost.W_e = np.block([[Qr,np.zeros((3,4))],
                                 [np.zeros((4,3)),Qa]
                                ])
        ocp.cost.yref_e =  np.zeros((7,1))


        max_thrust = 4.0 * self.max_thrust_per_motor
        max_torque_x_y = 2.0 * self.max_thrust_per_motor * self.arm_length
        gamma = self.km/self.kf
        max_torque_z = 4.0 * self.max_thrust_per_motor * gamma

        ocp.constraints.x0 = np.zeros(nx)
        ocp.constraints.lbu = np.array([-max_thrust, -max_torque_x_y, -max_torque_x_y, -max_torque_z])
        ocp.constraints.ubu = np.array([max_thrust, max_torque_x_y, max_torque_x_y, max_torque_z])
        ocp.constraints.idxbu = np.array([0,1,2,3], dtype=int)   
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.nlp_solver_type = 'SQP'
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        
        return ocp

    def make_trajectory(self, traj_type, params):
        """
        Multi-dimensional cubic/quintic trajectory generator.

        Parameters
        ----------
        traj_type : str
            'cubic' or 'quintic'
        params : dict
            't' : [t0, tf]
            'q' : [[q0_x, q0_y, q0_z], [qf_x, qf_y, qf_z]]   (initial and final positions)
            'v' : [[v0_x, v0_y, v0_z], [vf_x, vf_y, vf_z]]   (initial and final velocities)
            'a' : [[a0_x, a0_y, a0_z], [af_x, af_y, af_z]]   (only for quintic)
            'dt': time step

        Returns
        -------
        traj : dict
            't' : (N,) time array
            'q' : (N, D) positions
            'v' : (N, D) velocities
            'a' : (N, D) accelerations
        """
        t0, tf = params['t']
        q0, qf = np.array(params['q'][0]), np.array(params['q'][1])
        v0, vf = np.array(params['v'][0]), np.array(params['v'][1])
        dt = params['dt']
        D = len(q0)   # number of dimensions (1D, 2D, 3D, ...)

        if traj_type == 'quintic':
            a0, af = np.array(params['a'][0]), np.array(params['a'][1])

        times = np.arange(t0, tf+dt, dt)
        N = len(times)

        q_traj = np.zeros((N, D))
        v_traj = np.zeros((N, D))
        a_traj = np.zeros((N, D))

        # Solve trajectory for each dimension independently
        for d in range(D):
            if traj_type == 'cubic':
                A = np.array([
                    [1, t0, t0**2, t0**3],
                    [0, 1, 2*t0, 3*t0**2],
                    [1, tf, tf**2, tf**3],
                    [0, 1, 2*tf, 3*tf**2]
                ])
                b = np.array([q0[d], v0[d], qf[d], vf[d]])
                coeffs = np.linalg.solve(A, b)

                for i, t in enumerate(times):
                    q_traj[i, d] = coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3
                    v_traj[i, d] = coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2
                    a_traj[i, d] = 2*coeffs[2] + 6*coeffs[3]*t

            elif traj_type == 'quintic':
                A = np.array([
                    [1, t0, t0**2, t0**3, t0**4, t0**5],
                    [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                    [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                    [1, tf, tf**2, tf**3, tf**4, tf**5],
                    [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                    [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]
                ])
                b = np.array([q0[d], v0[d], a0[d], qf[d], vf[d], af[d]])
                coeffs = np.linalg.solve(A, b)

                for i, t in enumerate(times):
                    q_traj[i, d] = (coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 +
                                    coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5)
                    v_traj[i, d] = (coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 +
                                    4*coeffs[4]*t**3 + 5*coeffs[5]*t**4)
                    a_traj[i, d] = (2*coeffs[2] + 6*coeffs[3]*t +
                                    12*coeffs[4]*t**2 + 20*coeffs[5]*t**3)

        return {'t': times, 'q': q_traj, 'v': v_traj, 'a': a_traj}


    def plot_traj(self,time,x_traj):
        fig, axs = plt.subplots(2, 2, figsize=(10, 12), sharex=True)
        fig.suptitle('State vs Time')
        
        time = time[0:-1]
        axs[0][0].plot(time, x_traj[:, 0], label='x')
        axs[0][0].plot(time, x_traj[:, 1], label='y')
        axs[0][0].plot(time, x_traj[:, 2], label='z')
        axs[0][0].set_ylabel('Position')
        axs[0][0].grid(True)
        axs[0][0].legend()


        axs[0][1].plot(time, x_traj[:, 3], label='vx')
        axs[0][1].plot(time, x_traj[:, 4], label='vy')
        axs[0][1].plot(time, x_traj[:, 5] , label='vz')
        axs[0][1].set_ylabel('Velocity')
        axs[0][1].grid(True)
        axs[0][1].legend()

        axs[1][0].plot(time, x_traj[:, 6], label='qw')
        axs[1][0].plot(time, x_traj[:, 7], label='qx')
        axs[1][0].plot(time, x_traj[:, 8], label='qy')
        axs[1][0].plot(time, x_traj[:, 9], label='qy')
        axs[1][0].set_ylabel('Attitude')
        axs[1][0].grid(True)
        axs[1][0].legend()


        axs[1][1].plot(time, x_traj[:, 10], label='wx')
        axs[1][1].plot(time, x_traj[:,11], label='wy')
        axs[1][1].plot(time, x_traj[:,12], label='wz')
        axs[1][1].set_ylabel('Ang. Velocity')
        axs[1][1].set_xlabel('Time [s]')
        axs[1][1].grid(True)
        axs[1][1].legend()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_states_v_time(self):
        self.plot_traj(self.t,self.simX1)

# Main simulation loop
if __name__ == "__main__":
    # Initialize controller and MPC
    cf_controller = CrazyflieController()
    mpc = MPC(target_height=0.5)
    height = []
    # Simulation loop
    for step in range(mpc.sim_steps):
        # Get current state
        current_state = cf_controller.get_state()
        current_state = np.concatenate([current_state['position'].ravel(), current_state['velocity'].ravel(), current_state['orientation'].ravel(), current_state['angular_velocity'].ravel()])
        mpc.simX1[step,:] = current_state
        
        mpc_output = mpc.solve(step, current_state)

        # Apply control and step simulation
        new_state = cf_controller.step_simulation(mpc_output, mpc.trajectory1[step+1])
        
        # Print state every 100 steps
        if step % 10 == 0:
            print(f"Step {step}: Height = {new_state['position'][2]:.3f}m")
        height.append(new_state['position'][2])
    current_state = cf_controller.get_state()
    current_state = np.concatenate([current_state['position'].ravel(), current_state['velocity'].ravel(), current_state['orientation'].ravel(), current_state['angular_velocity'].ravel()])

    mpc.simX1[step,:] = current_state
    
    # Plotting obstacles and drone trajectories on a 3D plot
    traj_plot1 = mpc.trajectory1[:mpc.sim_steps+1, :3]
    x_obs = 0.2
    y_obs = 0.2
    z_obs = 0.4
    r_obs = 0.1
    margin = 0.01
    r_safe = r_obs + margin

    # Plot all 3 x, y, and z on a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Label axes
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Quadrotor Trajectory')

    # ax.set_zlim(0, 1)

    # Plot the reference trajectory
    ax.plot(traj_plot1[:, 0], traj_plot1[:, 1], traj_plot1[:, 2], '--', label='Reference Trajectory for 1', color='r')

    ## Uncomment below for animation of drone trajectories
    line1, = ax.plot([], [], [], 'b', label='Drone 1')

    # Animation loop
    for t in range(len(mpc.simX1)):
        line1.set_data(mpc.simX1[:t+1, 0], mpc.simX1[:t+1, 1])
        line1.set_3d_properties(mpc.simX1[:t+1, 2])
        plt.draw()
        plt.pause(0.01)  # adjust speed

    ax.legend()
    ax.grid(True)
    plt.show()
    
    p.disconnect()
    print(mpc.simX1[:,0:3])
    np.savetxt('height.csv', height, delimiter=',')

    mpc.plot_states_v_time()
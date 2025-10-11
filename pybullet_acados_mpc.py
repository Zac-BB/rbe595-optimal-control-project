import pybullet as p
import pybullet_data
import numpy as np
import scipy.linalg
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
        # p.applyExternalForce(self.cf_id,-1,[0.000008,0,0],[0,0,0],p.WORLD_FRAME)

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
    def __init__(self,target_height=None):
        print("init")


        self.Qr = np.diag([1000,1000,1000])
        self.Qv = np.diag([100,100,100])
        self.Qq = np.diag([100,100,100,100])
        self.Qw = np.diag([1000,1000,1000])
        self.R = np.diag([0.1,10,10,10])
        self.Pr = 3*self.Qr
        self.Pv = self.Qv
        self.Pq = self.Qq
        self.Pw = self.Qw 
        self.N  = 20 
        self.dt = 0.1   
        
        self.traj_time = 2.0

        self.mass = 0.027 # 0.027  # kg
        self.arm_length = 0.0397  # m
        self.kf = 3.16e-10  # thrust coefficient
        self.km = 7.94e-12  # moment coefficient
        self.max_rpm = 21000  # typical max RPM for CF2.0
        omega_max = self.max_rpm * 2*np.pi / 60.0  # rad/s
        self.max_thrust_per_motor = self.kf * (omega_max ** 2)
        self.solve_times = []

        self.target_height = target_height
        self.sim_steps = int(self.traj_time / self.dt)
        if target_height:
            params  = {
                't'  : [0, 1],
                'q'  : [[0, 0, 0], [0,0,target_height]],
                'v'  : [[0, 0, 0], [0, 0, 0]],
                'a'  : [[0, 0, 0], [0, 0, 0]],
                'dt' : self.dt
            } 
            takeoff = self.make_trajectory('quintic', params)
            circle = lambda t,dt: np.array([np.cos(t)-1,np.sin(t),target_height,-np.sin(t)*dt,np.cos(t)*dt,0])
            params  = {
                't'  : [0, 10],
                'q'  : [[0], [2*3.141529]],
                'v'  : [[0], [0]],
                'a'  : [[0], [0]],
                'dt' : self.dt
            } 
            quintic_space  = self.make_trajectory('quintic', params)
            circle_traj = np.vstack([circle(q,v)  for q, v in zip(quintic_space['q'].ravel(), quintic_space['v'].ravel())])
            pos_traj = np.vstack([np.hstack([takeoff['q'], takeoff['v']]), circle_traj])

            self.sim_steps = pos_traj.shape[0]-1
            attitude = np.hstack((np.ones((self.sim_steps+1,1)),np.zeros((self.sim_steps+1,6)) ))
            self.trajectory1 = np.concat([pos_traj, attitude], axis=1)
        else:

            params  = {
                't'  : [0, 10],
                'q'  : [[0, 0, 0], [1, 1, 1]],
                'v'  : [[0, 0, 0], [0, 0, 0]],
                'a'  : [[0, 0, 0], [0, 0, 0]],
                'dt' : self.dt
            } 
            pos_traj = self.make_trajectory('quintic', params)
            
            self.sim_steps = pos_traj['t'].shape[0]-1
            attitude = np.hstack((np.ones((self.sim_steps+1,1)),np.zeros((self.sim_steps+1,6)) ))
            self.trajectory1 = np.concat([pos_traj['q'], pos_traj['v'], attitude], axis=1)
        
        self.simX1 = np.zeros((self.sim_steps,13))
        self.ocp = self.create_ocp()
        self.ocp_solver = AcadosOcpSolver(self.ocp)


        self.ocp = self.create_ocp()
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


        
        self.ocp_solver.set(0, "lbx", current_state)
        self.ocp_solver.set(0, "ubx", current_state)
        self.ocp_solver.set(0, "x", current_state)

        get_idx = lambda i : min(step+i,self.sim_steps)
        for i in range(1, self.N):
            y_ref_i = np.concat([self.trajectory1[get_idx(i), :],[self.mass*9.81] ,np.zeros((3,))])
            self.ocp_solver.cost_set(i, 'yref', y_ref_i) 

        
        self.ocp_solver.cost_set(self.N, 'yref', np.concat([self.trajectory1[get_idx(self.N), 0:3],self.trajectory1[get_idx(self.N), 6:10]]))

        status = self.ocp_solver.solve()
        if status != 0:
            print(f"[MPC] Solver failed with status: {status}")
        u = self.ocp_solver.get(0, "u")   

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
        ny = nx + nu

        ocp.solver_options.N_horizon = self.N
        
        Q = scipy.linalg.block_diag(self.Qr, self.Qv, self.Qq, self.Qw)
        
        R = self.R
        ocp.cost.cost_type = "LINEAR_LS"
        Vx = np.block([[np.eye(nx)],[np.zeros((nu,nx))]])
        
        Vu = np.block([[np.zeros((nx,nu))],[np.eye(nu)]])
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.yref = np.zeros((ny, ))

        ocp.cost.cost_type_e = "LINEAR_LS"
        Vxe = np.block([[np.eye(3),np.zeros((3,10))],[np.zeros((4,6)),np.eye(4),np.zeros((4,3))]])
        ocp.cost.Vx_e = Vxe
        P = scipy.linalg.block_diag(self.Pr, self.Pq)
        ocp.cost.W_e = P
        ocp.cost.yref_e = np.zeros((7, ))
        
        
        
        ocp.solver_options.qp_solver        = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.integrator_type  = 'ERK'
        ocp.solver_options.nlp_solver_type  = 'SQP'
        ocp.solver_options.tf               = self.N*self.dt


        umax = np.array([0.7358, 0.0292, 0.0292, 0.0185])
        umin = np.array([0, -0.0292, -0.0292, -0.0185])
        
        ocp.constraints.x0 = np.zeros((nx, ))
        ocp.constraints.lbu = umin
        ocp.constraints.ubu = umax
        ocp.constraints.idxbu = np.array(range(nu))
        
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

    def analyze(self):
        t = np.linspace(0,self.traj_time,self.simX1.shape[0])
        pred = self.trajectory1[1:,:]
        actual = self.simX1
        error = actual- pred
        sum_error = np.diag(error.T@error)
        sum_error = sum_error/actual.shape[0]

        rmse = np.sqrt(sum_error)
        print("rmse: {}".format(rmse))
        fig, axs = plt.subplots(4,1, figsize=(10, 12), sharex=True)
        fig.suptitle('State vs Time')
        
        time = t
        axs[0].plot(time, actual[:, 0], label='x',color='tab:blue')
        axs[0].plot(time, actual[:, 1], label='y',color='tab:orange')
        axs[0].plot(time, actual[:, 2], label='z',color='tab:green')
        axs[0].plot(time, pred[:,0],label='x_ref',color='tab:blue',linestyle='dashed')
        axs[0].plot(time, pred[:,1],label='y_ref',color='tab:orange',linestyle='dashed')
        axs[0].plot(time, pred[:,2],label='z_ref',color='tab:green',linestyle='dashed')
        axs[0].set_ylabel('Position')
        axs[0].grid(True)
        axs[0].legend()


        axs[1].plot(time, actual[:, 3], label='vx',color='tab:blue')
        axs[1].plot(time, actual[:, 4], label='vy',color='tab:orange')
        axs[1].plot(time, actual[:, 5] , label='vz',color='tab:green')
        axs[1].plot(time, pred[:,3],label='vx_ref',color='tab:blue',linestyle='dashed')
        axs[1].plot(time, pred[:,4],label='vy_ref',color='tab:orange',linestyle='dashed')
        axs[1].plot(time, pred[:,5],label='vz_ref',color='tab:green',linestyle='dashed')
        axs[1].set_ylabel('Velocity')
        axs[1].grid(True)
        axs[1].legend()

        axs[2].plot(time, actual[:, 6], label='qw',color='tab:blue')
        axs[2].plot(time, actual[:, 7], label='qx',color='tab:orange')
        axs[2].plot(time, actual[:, 8], label='qy',color='tab:green')
        axs[2].plot(time, actual[:, 9], label='qy',color='tab:red')
        axs[2].plot(time, pred[:,6],label='qw_ref',color='tab:blue',linestyle='dashed')
        axs[2].plot(time, pred[:,7],label='qx_ref',color='tab:orange',linestyle='dashed')
        axs[2].plot(time, pred[:,8],label='qy_ref',color='tab:green',linestyle='dashed')
        axs[2].plot(time, pred[:,9],label='qy_ref',color='tab:red',linestyle='dashed')
        axs[2].set_ylabel('Attitude')
        axs[2].grid(True)
        axs[2].legend()


        axs[3].plot(time, actual[:, 10], label='wx',color='tab:blue')
        axs[3].plot(time, actual[:,11], label='wy',color='tab:orange')
        axs[3].plot(time, actual[:,12], label='wz',color='tab:green')
        axs[3].plot(time, pred[:,10],label='wx_ref',color='tab:blue',linestyle='dashed')
        axs[3].plot(time, pred[:,11],label='wy_ref',color='tab:orange',linestyle='dashed')
        axs[3].plot(time, pred[:,12],label='wz_ref',color='tab:green',linestyle='dashed')
        axs[3].set_ylabel('Ang. Velocity')
        axs[3].set_xlabel('Time [s]')
        axs[3].grid(True)
        axs[3].legend()
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plt.figure()
        # with open("solveTimesSPI.txt", "a") as f:
        #     np.savetxt(f, self.solve_times)  
        loaded = self.solve_times #np.loadtxt("solveTimesSPI.txt")
        print(f'data points: {len(loaded)}')
        print(f'mean: {np.mean(loaded)}, standard Devation: {np.std(loaded)}')
        plt.hist(loaded, bins=30, color='skyblue', edgecolor='k')
        plt.xlabel("Solve Time [ms]")
        plt.ylabel("Frequency")
        plt.title("Histogram of NMPC Solve Times SPI ")
        plt.show()

        print("hello")

# Main simulation loop
if __name__ == "__main__":
    # Initialize controller and MPC
    cf_controller = CrazyflieController()
    mpc = MPC(target_height=None)
    height = []
    # Simulation loop
    for step in range(mpc.sim_steps):
        # Get current state
        current_state = cf_controller.get_state()
        current_state = np.concatenate([current_state['position'].ravel(), current_state['velocity'].ravel(), current_state['orientation'].ravel(), current_state['angular_velocity'].ravel()])
        mpc.simX1[step,:] = current_state
        start_time = time.perf_counter()
        mpc_output = mpc.solve(step, current_state)
        end_time = time.perf_counter()
        elapsed_timne = end_time - start_time
        mpc.solve_times.append(elapsed_timne*1000)
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
    # print(mpc.simX1[:,0:3])
    # np.savetxt('height.csv', height, delimiter=',')

    mpc.analyze()
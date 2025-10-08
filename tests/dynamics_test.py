import unittest
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from quadrotor_model import quadrotor_model_auto
class Test_Dynamics(unittest.TestCase):

    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.show_plots = False
    def test_straight_up(self):

        model = quadrotor_model_auto()
        x = model.x
        u = model.u
        xdot = model.f_expl_expr.T
        func = ca.Function('f', [x, u], [xdot])

        dt = 0.01  
        ode = {'x': x, 'p': u, 'ode': func(x, u)} 
        opts = {'tf': dt, 'reltol': 1e-8, 'abstol': 1e-8}
        integrator = ca.integrator('integrator', 'cvodes', ode, opts)


        T = 10.0       
        N = int(T/dt) 
        x0 = np.array([0.0, 0.0, 0.0
                       , 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,])  
        u_const = np.array([1,0,0,0])  


        x_traj = np.zeros((N+1, len(x0)))
        x_traj[0] = x0
        time = np.linspace(0, T, N+1)

        for k in range(N):
            res = integrator(x0=x_traj[k], p=u_const)
            x_traj[k+1] = np.array(res['xf']).squeeze()
        
        for i,xi in enumerate(x_traj):
            if i == 0:
                continue
            xim1 = x_traj[i-1,:]
            change = xi[5] - xim1[5]
            dx = change / dt
            
            baseline = -9.81 + 1/0.03
            # result = func(x0,u_const)
            self.assertTrue(abs(dx - baseline) < 1e-5)
            
        
        if (self.show_plots):
            fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            fig.suptitle('Dynamics under constant u1 = %.2f' % u_const[0])

            axs[0].plot(time, x_traj[:, 0], label='x')
            axs[0].plot(time, x_traj[:, 1], label='y')
            axs[0].plot(time, x_traj[:, 2], label='z')
            axs[0].set_ylabel('Position')
            axs[0].grid(True)
            axs[0].legend()


            axs[1].plot(time, x_traj[:, 3], label='vx')
            axs[1].plot(time, x_traj[:, 4], label='vy')
            axs[1].plot(time, x_traj[:, 5] , label='vz')
            axs[1].set_ylabel('Velocity')
            axs[1].grid(True)
            axs[1].legend()

            axs[2].plot(time, x_traj[:, 6], label='qw')
            axs[2].plot(time, x_traj[:, 7], label='qx')
            axs[2].plot(time, x_traj[:, 8], label='qy')
            axs[2].plot(time, x_traj[:, 9], label='qy')
            axs[2].set_ylabel('Position')
            axs[2].grid(True)
            axs[2].legend()


            axs[3].plot(time, x_traj[:, 10], label='wx')
            axs[3].plot(time, x_traj[:,11], label='wy')
            axs[3].plot(time, x_traj[:,12], label='wz')
            axs[3].set_ylabel('Ang. Velocity')
            axs[3].set_xlabel('Time [s]')
            axs[3].grid(True)
            axs[3].legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            self.assertTrue(True)






    def test_free_fall(self):
        model = quadrotor_model_auto()
        x = model.x
        u = model.u
        xdot = model.f_expl_expr.T
        func = ca.Function('f', [x, u], [xdot])

        dt = 0.01 
        ode = {'x': x, 'p': u, 'ode': func(x, u)}  
        opts = {'tf': dt, 'reltol': 1e-8, 'abstol': 1e-8}
        integrator = ca.integrator('integrator', 'cvodes', ode, opts)


        T = 10.0       
        N = int(T/dt)  
        x0 = np.array([0.0, 0.0, 0.0
                       , 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,]) 
        u_const = np.array([0,0,0,0])  


        x_traj = np.zeros((N+1, len(x0)))
        x_traj[0] = x0
        time = np.linspace(0, T, N+1)

        for k in range(N):
            res = integrator(x0=x_traj[k], p=u_const)
            x_traj[k+1] = np.array(res['xf']).squeeze()


        for i,xi in enumerate(x_traj):
            if i == 0:
                continue
            xim1 = x_traj[i-1,:]
            change = xi[5] - xim1[5]
            dx = change / dt
            
            baseline = -9.81 
            # result = func(x0,u_const)
            self.assertTrue(abs(dx - baseline) < 1e-5)
       
        
        if (self.show_plots):
            fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            fig.suptitle('Dynamics under constant u1 = %.2f' % u_const[0])

            axs[0].plot(time, x_traj[:, 0], label='x')
            axs[0].plot(time, x_traj[:, 1], label='y')
            axs[0].plot(time, x_traj[:, 2], label='z')
            axs[0].set_ylabel('Position')
            axs[0].grid(True)
            axs[0].legend()


            axs[1].plot(time, x_traj[:, 3], label='vx')
            axs[1].plot(time, x_traj[:, 4], label='vy')
            axs[1].plot(time, x_traj[:, 5] , label='vz')
            axs[1].set_ylabel('Velocity')
            axs[1].grid(True)
            axs[1].legend()

            axs[2].plot(time, x_traj[:, 6], label='qw')
            axs[2].plot(time, x_traj[:, 7], label='qx')
            axs[2].plot(time, x_traj[:, 8], label='qy')
            axs[2].plot(time, x_traj[:, 9], label='qy')
            axs[2].set_ylabel('Position')
            axs[2].grid(True)
            axs[2].legend()


            axs[3].plot(time, x_traj[:, 10], label='wx')
            axs[3].plot(time, x_traj[:,11], label='wy')
            axs[3].plot(time, x_traj[:,12], label='wz')
            axs[3].set_ylabel('Ang. Velocity')
            axs[3].set_xlabel('Time [s]')
            axs[3].grid(True)
            axs[3].legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            self.assertTrue(True)

    def test_rotation_x(self):

        model = quadrotor_model_auto()
        x = model.x
        u = model.u
        xdot = model.f_expl_expr.T
        func = ca.Function('f', [x, u], [xdot])

        dt = 0.01 
        ode = {'x': x, 'p': u, 'ode': func(x, u)}  
        opts = {'tf': dt, 'reltol': 1e-8, 'abstol': 1e-8}
        integrator = ca.integrator('integrator', 'cvodes', ode, opts)


        T = 10.0       
        N = int(T/dt)  
        x0 = np.array([0.0, 0.0, 0.0
                       , 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,]) 
        u_const = np.array([0,1,0,0])  


        x_traj = np.zeros((N+1, len(x0)))
        x_traj[0] = x0
        time = np.linspace(0, T, N+1)

        for k in range(N):
            res = integrator(x0=x_traj[k], p=u_const)
            x_traj[k+1] = np.array(res['xf']).squeeze()

        

        for i,xi in enumerate(x_traj):
            if i == 0:
                continue
            xim1 = x_traj[i-1,:]
            change = xi[10] - xim1[10]
            dx = change / dt
            
            baseline = 1/1.43
            # result = func(x0,u_const)
            self.assertTrue(abs(dx - baseline) < 1e-5)
        if (self.show_plots):
            fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            fig.suptitle('Dynamics under constant u1 = %.2f' % u_const[0])

            axs[0].plot(time, x_traj[:, 0], label='x')
            axs[0].plot(time, x_traj[:, 1], label='y')
            axs[0].plot(time, x_traj[:, 2], label='z')
            axs[0].set_ylabel('Position')
            axs[0].grid(True)
            axs[0].legend()


            axs[1].plot(time, x_traj[:, 3], label='vx')
            axs[1].plot(time, x_traj[:, 4], label='vy')
            axs[1].plot(time, x_traj[:, 5] , label='vz')
            axs[1].set_ylabel('Velocity')
            axs[1].grid(True)
            axs[1].legend()

            axs[2].plot(time, x_traj[:, 6], label='qw')
            axs[2].plot(time, x_traj[:, 7], label='qx')
            axs[2].plot(time, x_traj[:, 8], label='qy')
            axs[2].plot(time, x_traj[:, 9], label='qy')
            axs[2].set_ylabel('Position')
            axs[2].grid(True)
            axs[2].legend()


            axs[3].plot(time, x_traj[:, 10], label='wx')
            axs[3].plot(time, x_traj[:,11], label='wy')
            axs[3].plot(time, x_traj[:,12], label='wz')
            axs[3].set_ylabel('Ang. Velocity')
            axs[3].set_xlabel('Time [s]')
            axs[3].grid(True)
            axs[3].legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            self.assertTrue(True)

    def test_rotation_y(self):

        model = quadrotor_model_auto()
        x = model.x
        u = model.u
        xdot = model.f_expl_expr.T
        func = ca.Function('f', [x, u], [xdot])

        dt = 0.01 
        ode = {'x': x, 'p': u, 'ode': func(x, u)}  
        opts = {'tf': dt, 'reltol': 1e-8, 'abstol': 1e-8}
        integrator = ca.integrator('integrator', 'cvodes', ode, opts)


        T = 10.0       
        N = int(T/dt)  
        x0 = np.array([0.0, 0.0, 0.0
                       , 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,]) 
        u_const = np.array([0,0,1,0])  


        x_traj = np.zeros((N+1, len(x0)))
        x_traj[0] = x0
        time = np.linspace(0, T, N+1)

        for k in range(N):
            res = integrator(x0=x_traj[k], p=u_const)
            x_traj[k+1] = np.array(res['xf']).squeeze()

        

        for i,xi in enumerate(x_traj):
            if i == 0:
                continue
            xim1 = x_traj[i-1,:]
            change = xi[11] - xim1[11]
            dx = change / dt
            
            baseline = 1/1.43
            # result = func(x0,u_const)
            self.assertTrue(abs(dx - baseline) < 1e-5)
        if (self.show_plots):
            fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            fig.suptitle('Dynamics under constant u1 = %.2f' % u_const[0])

            axs[0].plot(time, x_traj[:, 0], label='x')
            axs[0].plot(time, x_traj[:, 1], label='y')
            axs[0].plot(time, x_traj[:, 2], label='z')
            axs[0].set_ylabel('Position')
            axs[0].grid(True)
            axs[0].legend()


            axs[1].plot(time, x_traj[:, 3], label='vx')
            axs[1].plot(time, x_traj[:, 4], label='vy')
            axs[1].plot(time, x_traj[:, 5] , label='vz')
            axs[1].set_ylabel('Velocity')
            axs[1].grid(True)
            axs[1].legend()

            axs[2].plot(time, x_traj[:, 6], label='qw')
            axs[2].plot(time, x_traj[:, 7], label='qx')
            axs[2].plot(time, x_traj[:, 8], label='qy')
            axs[2].plot(time, x_traj[:, 9], label='qy')
            axs[2].set_ylabel('Position')
            axs[2].grid(True)
            axs[2].legend()


            axs[3].plot(time, x_traj[:, 10], label='wx')
            axs[3].plot(time, x_traj[:,11], label='wy')
            axs[3].plot(time, x_traj[:,12], label='wz')
            axs[3].set_ylabel('Ang. Velocity')
            axs[3].set_xlabel('Time [s]')
            axs[3].grid(True)
            axs[3].legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            self.assertTrue(True)

    def test_rotation_z(self):

        model = quadrotor_model_auto()
        x = model.x
        u = model.u
        xdot = model.f_expl_expr.T
        func = ca.Function('f', [x, u], [xdot])

        dt = 0.01 
        ode = {'x': x, 'p': u, 'ode': func(x, u)}  
        opts = {'tf': dt, 'reltol': 1e-8, 'abstol': 1e-8}
        integrator = ca.integrator('integrator', 'cvodes', ode, opts)


        T = 10.0       
        N = int(T/dt)  
        x0 = np.array([0.0, 0.0, 0.0
                       , 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,]) 
        u_const = np.array([0,0,0,1])  


        x_traj = np.zeros((N+1, len(x0)))
        x_traj[0] = x0
        time = np.linspace(0, T, N+1)

        for k in range(N):
            res = integrator(x0=x_traj[k], p=u_const)
            x_traj[k+1] = np.array(res['xf']).squeeze()

        

        for i,xi in enumerate(x_traj):
            if i == 0:
                continue
            xim1 = x_traj[i-1,:]
            change = xi[12] - xim1[12]
            dx = change / dt
            
            baseline = 1/2.89
            # result = func(x0,u_const)
            self.assertTrue(abs(dx - baseline) < 1e-5)
        if (self.show_plots):
            fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            fig.suptitle('Dynamics under constant u1 = %.2f' % u_const[0])

            axs[0].plot(time, x_traj[:, 0], label='x')
            axs[0].plot(time, x_traj[:, 1], label='y')
            axs[0].plot(time, x_traj[:, 2], label='z')
            axs[0].set_ylabel('Position')
            axs[0].grid(True)
            axs[0].legend()


            axs[1].plot(time, x_traj[:, 3], label='vx')
            axs[1].plot(time, x_traj[:, 4], label='vy')
            axs[1].plot(time, x_traj[:, 5] , label='vz')
            axs[1].set_ylabel('Velocity')
            axs[1].grid(True)
            axs[1].legend()

            axs[2].plot(time, x_traj[:, 6], label='qw')
            axs[2].plot(time, x_traj[:, 7], label='qx')
            axs[2].plot(time, x_traj[:, 8], label='qy')
            axs[2].plot(time, x_traj[:, 9], label='qy')
            axs[2].set_ylabel('Position')
            axs[2].grid(True)
            axs[2].legend()


            axs[3].plot(time, x_traj[:, 10], label='wx')
            axs[3].plot(time, x_traj[:,11], label='wy')
            axs[3].plot(time, x_traj[:,12], label='wz')
            axs[3].set_ylabel('Ang. Velocity')
            axs[3].set_xlabel('Time [s]')
            axs[3].grid(True)
            axs[3].legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
            self.assertTrue(True)
if __name__ == '__main__':
    unittest.main()
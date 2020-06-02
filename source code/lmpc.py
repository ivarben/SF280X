import numpy as np
import scipy.linalg as la
import time
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.path as mpath

class LMPC:
    def __init__(self, T, dt, N, J, R):
        self.road_min_speed = 0
        self.road_max_speed = 31
        self.v_L = 25 # velocity of vehicle L
        self.v_P = 28 # initial (preferred) velocity of E
        self.w_C, self.l_C = 1.8, 4.5 # width, length of vehicles (specifically vehicle L)
        self.w_L = 3.25 # width of lane
        self.time_gap = 2 # time gap according to 2 s rule
        self.L = self.v_P * self.time_gap + self.l_C # safe distance infront and behind
        self.d = self.v_P * (1+self.time_gap) + self.l_C # distance at which overtaking starts
        self.tol = 0.001

        self.dt = dt # time step
        self.A = np.array([[1,0,self.dt,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]])
        self.B = np.array([[0,0],[0,0],[dt,0],[0,dt]])
        self.n, self.m = self.A.shape[0], self.B.shape[1] # dimensions

        self.R = R # quadratic costs
        self.r = R[1,1] # size of diagonal elements

        self.xS = np.array([-self.d, 0, self.v_P - self.v_L, 0])
        self.xF = np.array([self.d, 0, self.v_P - self.v_L, 0])

        self.time = np.arange(T+1)
        self.dt = dt
        self.completion = [0] * (J+1) # index at which xF was reached during each iteration
        self.N = N # Control/ prediction horizon of each MPC problem
        self.J = J # Number of iterations of LMPC
        self.comp_time = np.zeros((1, T+1, J+1)) # store computational time of solver at each time step and iteration
        self.avgcomptime = np.zeros(J)
        # data which needs to be stored:
        self.stage_costs = np.zeros((1, T+1, J+1)) # realized stage costs over time, saves all iterations. Used for ctg.
        self.iteration_costs = np.zeros(J+1) # total cost for each iteration
        self.SS = np.empty([1+self.n, 0]) # Safe set

        self.x_trajectory = np.zeros((self.n, T+1, J+1)) # realized state trajectories (saves all iterations, even 0)
        self.u_trajectory = np.zeros((self.m, T+1, J+1)) # realized control trajectories (saves all iterations, even 0)
        self.all_predictions = np.zeros((self.n, self.N+1, T+1, J+1)) # saves all predictions

    def update(self, x, u):
        return self.A@x + self.B@u

    def solve(self):
        self.iteration_0()
        self.iteration_costs[0] = np.sum(self.stage_costs[0,:,0])

        for j in range(1, self.J+1): # Iteration number j
            print('Iteration: ' + str(j))
            self.iterate(j)
            self.iteration_costs[j] = np.sum(self.stage_costs[0,:self.completion[j]+1,j])
            self.avgcomptime[j-1] = np.mean(self.comp_time[0,:self.completion[j]+1, j])

    def iteration_0(self):
        self.x_trajectory[:,0,0] = self.xS
        self.u_trajectory[:,0,0] = np.array([0, 0.9]) # hardcoded acceleration
        self.stage_costs[:,0,0] = self.evaluate_stage_cost(self.x_trajectory[:,0,0], self.u_trajectory[:,0,0])

        for t in self.time[1:]: # iterate time
            self.x_trajectory[:,t,0] = self.update(self.x_trajectory[:,t-1,0], self.u_trajectory[:,t-1,0]) # Dynamics

            if t >= 1 and t <= 3:
                self.u_trajectory[:,t,0] = np.array([0, 0.9])
            elif t >= 4 and t <= 7:
                self.u_trajectory[:,t,0] = np.array([0, -0.9])
            elif t >= 8 and t <= 12:
                self.u_trajectory[:,t,0] = np.array([0.8, 0])
            elif t >= 73 and t <= 74:
                self.u_trajectory[:,t,0] = np.array([-2, 0])
            elif t >= 75 and t <= 78:
                self.u_trajectory[:,t,0] = np.array([0, -0.9])
            elif t >= 79 and t <= 82:
                self.u_trajectory[:,t,0] = np.array([0, 0.9])

            self.stage_costs[0,t,0] = self.evaluate_stage_cost(self.x_trajectory[:,t,0], self.u_trajectory[:,t,0])

        self.completion[0] = self.time[-1]
        self.append_SS(0)

    def iterate(self, j):
        self.solve_FHOCP(self.xS) # t = 0
        self.apply(j, 0)

        for t in self.time[1:]: # t = 1, 2, ...
            if self.x_trajectory[0,t-1,j] >= self.xF[0] and (self.x_trajectory[1,t-1,j] - self.xF[1])**2 < self.tol and (self.x_trajectory[2,t-1,j] - self.xF[2])**2 < self.tol and (self.x_trajectory[3,t-1,j] - self.xF[3])**2 < self.tol:
                break
            else:
                self.completion[j] += 1
                self.solve_FHOCP(self.update(self.x_trajectory[:,t-1,j], self.u_trajectory[:,t-1,j]))
                self.apply(j, t)

        self.append_SS(j)

    def apply(self, j, t):
        ### Apply the optimized controls
        self.all_predictions[:,:,t,j] = self.x_star
        self.x_trajectory[:,t,j] = self.x_star[:,0]
        self.u_trajectory[:,t,j] = self.u_star[:,0]
        self.stage_costs[:,t,j] = self.evaluate_stage_cost(self.x_trajectory[:,t,j], self.u_trajectory[:,t,j])
        self.comp_time[0,t,j] = self.current_comp_time

    def plot(self):
        fig = plt.figure()
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
        ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
        vehicle_L = Rectangle((-self.l_C/2, -self.w_C/2), self.l_C, self.w_C, label = 'Vehicle L', color = 'r')
        ax1.add_artist(vehicle_L)
        #ax1.set_aspect('equal')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax2.set_xlabel('time step')
        ax2.set_ylabel('x dot (m/s)')
        ax3.set_xlabel('time step')
        ax3.set_ylabel('y dot (m/s)')

        # road sides
        ax1.axhline(-0.5*self.w_L, lw = 2, color = '0.5')
        ax1.axhline(1.5*self.w_L, lw = 2, color = '0.5')
        ax1.axhline(0.5*self.w_L, ls = '--', lw = 2, color = '0.7')

        # ymin
        ax1.plot([-self.d, -self.L], [-0.5*self.w_L + 0.5*self.w_C] * 2, lw = 1, color = 'r')
        ax1.plot([-self.L, self.L], [0.5*self.w_L + 0.5*self.w_C] * 2, lw = 1, color = 'r')
        ax1.plot([self.L, self.d], [-0.5*self.w_L + 0.5*self.w_C] * 2, lw = 1, color = 'r')
        # ymax
        ax1.plot([-self.d, self.d], [1.5*self.w_L - 0.5*self.w_C] * 2, lw = 1, color = 'r')

        for j in range(self.J+1):
            ax1.plot(self.x_trajectory[0,:self.completion[j]+1,j], self.x_trajectory[1,:self.completion[j]+1,j], marker = '.' , markersize = 4, lw = 0.75, label = ' iter ' + str(j))
            ax2.plot(self.time[:self.completion[j]+1], self.x_trajectory[2,:self.completion[j]+1,j], lw = 0.75, label = ' iter ' + str(j))
            ax3.plot(self.time[:self.completion[j]+1], self.x_trajectory[3,:self.completion[j]+1,j], lw = 0.75, label = ' iter ' + str(j))

        handles, labels = ax3.get_legend_handles_labels()
        fig.legend(handles, labels, loc='right')
        fig.tight_layout()
        fig.set_size_inches(11, 7)
        fig.savefig('plots/' + 'TRAJ_' + self.name + '_N=' + str(self.N) + '_r=' + str(self.r) + '_J=' + str(self.J) + '.pdf', bbox_inches='tight', dpi = 500)
        plt.close()

        fig, ax = plt.subplots()
        ax.set_xlabel('time step')
        ax.set_ylabel('computational time (s)')
        for j in range(self.J+1):
            ax.plot(self.time, self.comp_time[0,:,j])
        fig.set_size_inches(11, 7)
        fig.savefig('plots/' + 'COMPT_' + self.name + '_N=' + str(self.N) + '_r=' + str(self.r) + '_J=' + str(self.J) + '.pdf', bbox_inches='tight', dpi = 500)
        plt.close()

    def animate(self):
        for j in range(1,self.J+1):
            self.create_fig(j)

    def create_fig(self, j):
        fig, ax = plt.subplots()
        lineE, = ax.plot([], [], 'o', lw=3, color = 'b')
        lineP, = ax.plot([], [], '.', lw=3, color = '0.1')
        ax.set_xlim(-100, 120)
        ax.set_ylim(-5, 10)
        for j2 in range(j):
            ax.plot(self.x_trajectory[0,:self.completion[j2]+1,j2], self.x_trajectory[1,:self.completion[j2]+1,j2], '.')
        ax.axhline(-0.5*self.w_L, lw = 2, color = '0.5')
        ax.axhline(1.5*self.w_L, lw = 2, color = '0.5')
        ax.axhline(0.5*self.w_L, ls = '--', lw = 2, color = '0.7')
        ax.plot([-self.d, -self.L], [-0.5*self.w_L + 0.5*self.w_C] * 2, lw = 1, color = 'r')
        ax.plot([-self.L, self.L], [0.5*self.w_L + 0.5*self.w_C] * 2, lw = 1, color = 'r')
        ax.plot([self.L, self.d], [-0.5*self.w_L + 0.5*self.w_C] * 2, lw = 1, color = 'r')
        ax.plot([-self.d, self.d], [1.5*self.w_L - 0.5*self.w_C] * 2, lw = 1, color = 'r')
        vehicle_L = Rectangle((-self.l_C/2, -self.w_C/2), self.l_C, self.w_C, label = 'Vehicle L', color = 'r')
        ax.add_artist(vehicle_L)

        def init():
            lineE.set_data([], [])
            lineP.set_data([], [])
            return lineE, lineP,
        def update_animation(i, j):
            x = self.x_trajectory[0,[i],j]
            y = self.x_trajectory[1,[i],j]
            lineE.set_data(x, y)
            xp = self.all_predictions[0,:,i,j]
            yp = self.all_predictions[1,:,i,j]
            lineP.set_data(xp, yp)
            if i >= self.completion[j]-1:
                plt.close(lineE.axes.figure)
            return lineE, lineP,

        anim = FuncAnimation(fig, update_animation, init_func=init, fargs = (j,), frames=self.completion[1], interval=50, blit=True, repeat = False)
        plt.show()

    def save(self):
        np.savez('npfiles/' + self.name + '_N=' + str(self.N) + '_J=' + str(self.J) + '_r=' + str(self.r),
        self.x_trajectory, self.u_trajectory, self.iteration_costs, self.stage_costs,
        self.all_predictions, self.comp_time, self.avgcomptime)

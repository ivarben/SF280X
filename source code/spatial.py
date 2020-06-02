from lmpc import *

class Spatial(LMPC):
    def __init__(self, T, dt, N, J, R):
        super().__init__(T, dt, N, J, R)
        self.name = 'spatial'
        self.eps = self.v_P - self.v_L # minimum speed
        self.dx = 1.5
        self.v_R = self.eps # reference speed to linearize around
        self.A = np.array([[1,0,0,0],[0,1,0,self.dx],[0,0,1,0],[0,0,0,1]])
        self.B = np.array([[self.dx,0,0],[0,0,0],[0,self.dx,0],[0,0,self.dx]])
        self.n, self.m = self.A.shape[0], self.B.shape[1] # dimensions
        self.R = np.array([[0,0,0], [0,self.R[0,0],0], [0,0,self.R[1,1]]])

        self.distance = np.arange(82 + (self.xF[0] - self.xS[0])/self.dx).astype(int)
        self.step_limit = 130 # see notes
        self.SS = np.empty([1+self.n, 0]) # Safe set
        self.comp_time = np.zeros((1, self.distance.shape[0], J+1))
        self.stage_costs = np.zeros((1, self.distance.shape[0], J+1)) # realized stage costs over time, saves all iterations. Used for ctg.
        self.x_trajectory = np.zeros((self.n, self.distance.shape[0], J+1)) # realized state trajectories (saves all iterations, even 0)
        self.u_trajectory = np.zeros((self.m, self.distance.shape[0], J+1)) # realized control trajectories (saves all iterations, even 0)
        self.all_predictions = np.zeros((self.n, self.N+1, self.distance.shape[0], J+1)) # saves all predictions

    def evaluate_stage_cost(self, x, u):
        return u.T@self.R@u

    def iteration_0(self):
        self.x_trajectory[:,0,0] = self.xS
        self.u_trajectory[:,0,0] = np.array([1, 0, 0.18]) # hardcoded input
        self.stage_costs[:,0,0] = self.evaluate_stage_cost(self.x_trajectory[:,0,0], self.u_trajectory[:,0,0])

        for d in self.distance[1:]: # iterate distance
            self.x_trajectory[:,d,0] = self.update(self.x_trajectory[:,d-1,0], self.u_trajectory[:,d-1,0]) # Dynamics
            self.u_trajectory[0,d,0] = 1

            if d <= 2:
                self.u_trajectory[1:,d,0] = np.array([0, 0.18])
            elif d >= 3 and d <= 5:
                self.u_trajectory[1:,d,0] = np.array([0, -0.18])
            elif d >= 110 and d <= 112:
                self.u_trajectory[1:,d,0] = np.array([0, -0.18])
            elif d >= 113 and d <= 115:
                self.u_trajectory[1:,d,0] = np.array([0, 0.18])

            self.stage_costs[0,d,0] = self.evaluate_stage_cost(self.x_trajectory[:,d,0], self.u_trajectory[:,d,0])

        self.completion[0] = self.distance[-1]
        self.append_SS(0)

    def iterate(self, j):
        self.solve_FHOCP(self.xS) # d = 0
        self.apply(j, 0)

        for d in self.distance[1:]: # d = 1, 2, ...
            if self.x_trajectory[0,d-1,j] - self.xF[0] >= 0 and (self.x_trajectory[1,d-1,j] - self.xF[1])**2 < self.tol and (self.x_trajectory[2,d-1,j] - self.xF[2])**2 < self.tol and (self.x_trajectory[3,d-1,j] - self.xF[3])**2 < self.tol:
                break
            else:
                self.completion[j] += 1
                self.solve_FHOCP(self.update(self.x_trajectory[:,d-1,j], self.u_trajectory[:,d-1,j]))
                self.apply(j, d)

        self.append_SS(j)

    def append_SS(self, j): # needs iteration number
        costs = self.stage_costs[:,:self.completion[j]+1,j]
        ctg = np.zeros(costs.shape)
        for i in range(ctg.shape[1]):
            ctg[0,i] = costs[0,i:].sum()
        self.SS = np.concatenate((self.SS, np.concatenate((ctg, self.x_trajectory[:,:self.completion[j]+1,j]), axis = 0)), axis = 1)

    def solve_FHOCP(self, x_0):
        model = gp.Model('prediction')
        model.setParam('OutputFlag', 0)
        model.setParam('Presolve', 0)
        model.setParam('IntFeasTol', 1e-8)
        #model.setParam('FeasibilityTol', 1e-8)

        SSx = self.SS[:, self.SS[1, :] >= min(x_0[0] + self.N*self.dx, self.xS[0] + self.step_limit*self.dx)]

        x = model.addMVar((self.n, self.N + 1), lb = -1000, name = 'x')
        u = model.addMVar((self.m, self.N), lb = -1000, name = 'u')
        delta = model.addMVar(SSx.shape[1], vtype = GRB.BINARY, name = 'delta')

        model.addConstr(x[:,0] == x_0) # initial condition
        model.addConstr(x[:,self.N] == SSx[1:,:]@delta) # terminal constraint
        model.addConstr(delta.sum() == 1)
        terminal_cost = SSx[0,:]@delta
        stage_costs = 0

        for k in range(self.N):
            model.addConstr(x[1,k] <= 1.5*self.w_L - 0.5*self.w_C) # y max
            if x_0[0] + self.dx*k >= -self.L and x_0[0] + self.dx*k <= self.L:
                model.addConstr(x[1,k] >= 0.5*self.w_L + 0.5*self.w_C) # ymin box
            else:
                model.addConstr(x[1,k] >= -0.5*self.w_L + 0.5*self.w_C) # ymin otherwise

            model.addConstr(x[2,k] >= self.eps) # min x speed
            model.addConstr(x[2,k] <= self.road_max_speed - self.v_L) # max x speed
            model.addConstr(x[3,k] <=  0.17*(1 + (self.v_L/self.v_R)*(2 - x[2,k]/self.v_R))) # side slip
            model.addConstr(x[3,k] >= -0.17*(1 + (self.v_L/self.v_R)*(2 - x[2,k]/self.v_R)))
            model.addConstr(u[0,k] == 1) # distance logic
            model.addConstr(u[1,k] <=  1*(1/self.v_R)*(2 - x[2,k]/self.v_R)) # ax_max
            model.addConstr(u[1,k] >= -4*(1/self.v_R)*(2 - x[2,k]/self.v_R)) # ax_min
            model.addConstr(u[2,k] <=  2*(1/self.v_R)*(2 - x[2,k]/self.v_R)) # ay_max
            model.addConstr(u[2,k] >= -2*(1/self.v_R)*(2 - x[2,k]/self.v_R)) # ay_min

            model.addConstr(x[:,k+1] == self.A@x[:,k] + self.B@u[:,k]) # dynamics
            stage_costs += u[:,k]@self.R@u[:,k]
            if x_0[0] + self.dx*k >= self.xS[0] + self.step_limit*self.dx:
                model.addConstr(x[0,k] >= self.xF[0])
                model.addConstr(x[1,k] == self.xF[1])
                model.addConstr(x[2,k] == self.xF[2])
                model.addConstr(x[3,k] == self.xF[3])

        model.setObjective(terminal_cost + stage_costs, GRB.MINIMIZE)
        model.optimize()
        self.current_comp_time = model.Runtime

        self.x_star = x.x
        self.u_star = u.x

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
        ax2.set_xlabel('distance step')
        ax2.set_ylabel('x dot (m/s)')
        ax3.set_xlabel('distance step')
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
            ax2.plot(self.distance[:self.completion[j]+1], self.x_trajectory[2,:self.completion[j]+1,j], lw = 0.75, label = ' iter ' + str(j))
            ax3.plot(self.distance[:self.completion[j]+1], self.x_trajectory[3,:self.completion[j]+1,j], lw = 0.75, label = ' iter ' + str(j))

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
            ax.plot(self.distance, self.comp_time[0,:,j])
        fig.set_size_inches(11, 7)
        fig.savefig('plots/' + 'COMPT_' + self.name + '_N=' + str(self.N) + '_r=' + str(self.r) + '_J=' + str(self.J) + '.pdf', bbox_inches='tight', dpi = 500)
        plt.close()

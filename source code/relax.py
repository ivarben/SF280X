from lmpc import *

class Relax(LMPC):
    def __init__(self, T, dt, N, J, R):
        super().__init__(T, dt, N, J, R)
        self.name = 'relax'
        self.W = 0.5*self.w_L + 0.5*self.w_C # lateral safety distance for FCC and RCC
        self.const1, self.const2 = 10000, 10000 # costs on slack variables

    def evaluate_stage_cost(self, x, u):
        not_done = 1
        if x[0] >= self.xF[0] and (x[1] - self.xF[1])**2 < self.tol and (x[2] - self.xF[2])**2 < self.tol and (x[3] - self.xF[3])**2 < self.tol:
            not_done = 0
        #(x - self.xF).T@self.Q@(x - self.xF)
        return u.T@self.R@u + not_done*self.dt

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
        #model.setParam('MIPGap', 1e-10)
        model.setParam('IntFeasTol', 1e-8)
        #model.setParam('FeasibilityTol', 1e-4)

        x = model.addMVar((self.n, self.N + 1), lb = -100, name = 'x')
        u = model.addMVar((self.m, self.N), lb = -10, name = 'u')
        T = model.addMVar(self.N, vtype = GRB.BINARY, name = 'T')
        delta = model.addMVar(self.SS.shape[1], vtype = GRB.BINARY, name = 'delta')
        e_xf = model.addMVar(1, name = 'e_xf')
        e_yf = model.addMVar(self.N, lb = -1000, name = 'e_yf')
        e_f = model.addMVar(self.N, name = 'e_f')
        e_xr = model.addMVar(1, name = 'e_xr')
        e_r = model.addMVar(self.N, name = 'e_r')

        d = x_0[0] # initialized at each optimization cycle
        phi = max(1, abs(x_0[0])) # psi = 5 must be tuned, initialized at each optimization cycle

        model.addConstr(x[:,0] == x_0) # initial condition
        model.addConstr(x[:,self.N] == self.SS[1:,:]@delta) # terminal constraint
        model.addConstr(delta.sum() == 1)
        terminal_cost = self.SS[0,:]@delta
        stage_costs = 0

        for k in range(self.N):
            stage_costs += self.dt*T[k] + u[:,k]@self.R@u[:,k] + e_f[k]@e_f[k]*self.const1 + e_r[k]@e_r[k]*self.const2

            model.addConstr(x[1,k]/self.W - x[0,k]/self.L + d * e_xf + e_yf[k]/phi + e_f[k] >= 1) #FCC
            model.addConstr(x[1,k]/self.W + x[0,k]/self.L - d * e_xr + e_r[k] >= 1) #RCC
            model.addConstr(e_yf[k] == x[1,k] - self.W)

            model.addConstr(x[1,k] <= 1.5*self.w_L - 0.5*self.w_C) # y max
            model.addConstr(x[1,k] >= -0.5*self.w_L + 0.5*self.w_C) # ymin
            model.addConstr(x[2,k] >= self.road_min_speed - self.v_L) # min x speed
            model.addConstr(x[2,k] <= self.road_max_speed - self.v_L) # max x speed
            model.addConstr(x[3,k] <=  0.17*(x[2,k] + self.v_L)) # side slip
            model.addConstr(x[3,k] >= -0.17*(x[2,k] + self.v_L))
            model.addConstr(u[0,k] <=  1) # ax_max
            model.addConstr(u[0,k] >= -4) # ax_min
            model.addConstr(u[1,k] <=  2) # ay_max
            model.addConstr(u[1,k] >= -2) # ay_min

            model.addConstr(x[:,k+1] == self.A@x[:,k] + self.B@u[:,k]) # dynamics

            model.addConstr(200 * T[k] >= self.xF[0] - x[0,k]) # far enough?
            model.addConstr(100 * T[k] >= x[1,k]@x[1,k] - 2*self.xF[1]*x[1,k] + self.xF[1]*self.xF[1]) # central enough?
            model.addConstr(100 * T[k] >= x[2,k]@x[2,k] - 2*self.xF[2]*x[2,k] + self.xF[2]*self.xF[2]) # right speed?
            model.addConstr(100 * T[k] >= x[3,k]@x[3,k] - 2*self.xF[3]*x[3,k] + self.xF[3]*self.xF[3]) # still enough?

        model.setObjective(terminal_cost + stage_costs, GRB.MINIMIZE)
        model.optimize()
        self.current_comp_time = model.Runtime

        self.x_star = x.x
        self.u_star = u.x
        self.e_f = e_f.x
        self.e_r = e_r.x

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
        ax1.plot([-self.d, self.d], [-0.5*self.w_L + 0.5*self.w_C] * 2, lw = 1, color = 'r')
        ax1.plot([-self.d, self.d], [1.5*self.w_L - 0.5*self.w_C] * 2, lw = 1, color = 'r')

        # FCC, RCC
        ax1.plot([(-1.5*self.w_L+self.W)*self.L/self.W, (0.5*self.w_L+self.W)*self.L/self.W], [1.5*self.w_L, -0.5*self.w_L], lw = 1, ls = '--', color = '0.5')
        ax1.plot([(1.5*self.w_L-self.W)*self.L/self.W, -(0.5*self.w_L+self.W)*self.L/self.W], [1.5*self.w_L, -0.5*self.w_L], lw = 1, ls = '--', color = '0.5')

        for j in range(self.J+1):
            ax1.plot(self.x_trajectory[0,:self.completion[j]+1,j], self.x_trajectory[1,:self.completion[j]+1,j], marker = '.' , markersize = 4, lw = 0.75, label = 'j = ' + str(j))
            ax2.plot(self.time[:self.completion[j]+1], self.x_trajectory[2,:self.completion[j]+1,j], lw = 0.75, label = 'j = ' + str(j))
            ax3.plot(self.time[:self.completion[j]+1], self.x_trajectory[3,:self.completion[j]+1,j], lw = 0.75, label = 'j = ' + str(j))

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
            ax.plot(self.x_trajectory[0,:self.completion[j]+1,j2], self.x_trajectory[1,:self.completion[j]+1,j2], '.')
        ax.axhline(-0.5*self.w_L, lw = 2, color = '0.5')
        ax.axhline(1.5*self.w_L, lw = 2, color = '0.5')
        ax.axhline(0.5*self.w_L, ls = '--', lw = 2, color = '0.7')
        ax.plot([(-1.5*self.w_L+self.W)*self.L/self.W, (0.5*self.w_L+self.W)*self.L/self.W], [1.5*self.w_L, -0.5*self.w_L], lw = 1, ls = '--', color = '0.5')
        ax.plot([(1.5*self.w_L-self.W)*self.L/self.W, -(0.5*self.w_L+self.W)*self.L/self.W], [1.5*self.w_L, -0.5*self.w_L], lw = 1, ls = '--', color = '0.5')
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

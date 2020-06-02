from relax import *

class RelaxLimited(Relax):
    def __init__(self, T, dt, N, J, R):
        super().__init__(T, dt, N, J, R)
        self.name = 'relax_limited'
        self.time_limit = 90 # 45s
        self.SS = self.SS = np.empty([2+self.n, 0]) # Safe set

    def evaluate_stage_cost(self, x, u):
        return u.T@self.R@u

    def append_SS(self, j): # needs iteration number
        costs = self.stage_costs[:,:self.completion[j]+1,j]
        ctg = np.zeros(costs.shape)
        for i in range(ctg.shape[1]):
            ctg[0,i] = costs[0,i:].sum()
        time = self.time[:self.completion[j]+1].reshape(-1,1).T
        self.SS = np.concatenate((self.SS, np.concatenate((ctg, time, self.x_trajectory[:,:self.completion[j]+1,j]), axis = 0)), axis = 1)

    def iterate(self, j):
        self.solve_FHOCP(self.xS, 0) # t = 0
        self.apply(j, 0)

        for t in self.time[1:]: # t = 1, 2, ...
            if self.x_trajectory[0,t-1,j] >= self.xF[0] and (self.x_trajectory[1,t-1,j] - self.xF[1])**2 < self.tol and (self.x_trajectory[2,t-1,j] - self.xF[2])**2 < self.tol and (self.x_trajectory[3,t-1,j] - self.xF[3])**2 < self.tol:
                break
            else:
                self.completion[j] += 1
                self.solve_FHOCP(self.update(self.x_trajectory[:,t-1,j], self.u_trajectory[:,t-1,j]), t)
                self.apply(j, t)

        self.append_SS(j)

    def solve_FHOCP(self, x_0, t_0):
        model = gp.Model('prediction')
        model.setParam('OutputFlag', 0)
        model.setParam('Presolve', 0)
        model.setParam('IntFeasTol', 1e-8)

        SSt = self.SS[:, self.SS[1, :] >= min(t_0 + self.N, self.time_limit)]

        x = model.addMVar((self.n, self.N + 1), lb = -100, name = 'x')
        u = model.addMVar((self.m, self.N), lb = -10, name = 'u')
        delta = model.addMVar(SSt.shape[1], vtype = GRB.BINARY, name = 'delta')
        e_xf = model.addMVar(1, name = 'e_xf')
        e_yf = model.addMVar(self.N, lb = -1000, name = 'e_yf')
        e_f = model.addMVar(self.N, name = 'e_f')
        e_xr = model.addMVar(1, name = 'e_xr')
        e_r = model.addMVar(self.N, name = 'e_r')

        d = x_0[0] # initialized at each optimization cycle
        phi = max(1, abs(x_0[0])) # psi = 5 must be tuned, initialized at each optimization cycle

        model.addConstr(x[:,0] == x_0) # initial condition
        model.addConstr(x[:,self.N] == SSt[2:,:]@delta) # terminal constraint
        model.addConstr(delta.sum() == 1)
        terminal_cost = SSt[0,:]@delta
        stage_costs = 0

        for k in range(self.N):
            stage_costs += u[:,k]@self.R@u[:,k] + e_f[k]@e_f[k]*self.const1 + e_r[k]@e_r[k]*self.const2
            #stage_costs += T[k] + u[:,k]@self.R@u[:,k] + e_f[k]*self.const1 - e_r[k]*self.const2

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

            if t_0 + k >= self.time_limit:
                model.addConstr(x[0,k] >= self.xF[0])
                model.addConstr(x[1,k] == self.xF[1])
                model.addConstr(x[2,k] == self.xF[2])
                model.addConstr(x[3,k] == self.xF[3])

        model.setObjective(terminal_cost + stage_costs, GRB.MINIMIZE)
        model.optimize()
        self.current_comp_time = model.Runtime

        self.x_star = x.x
        self.u_star = u.x
        self.e_f = e_f.x
        self.e_r = e_r.x

from original import *

class OriginalLimited(Original):
    def __init__(self, T, dt, N, J, R):
        super().__init__(T, dt, N, J, R)
        self.time_limit = 90 # 45 s
        self.SS = self.SS = np.empty([2+self.n, 0]) # Safe set
        self.name = 'original_limited'

    def evaluate_stage_cost(self, x, u):
        #(x - self.xF).T@self.Q@(x - self.xF)
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
        model.setParam('NonConvex', 2)
        model.setParam('IntFeasTol', 1e-8)
        #model.setParam('FeasibilityTol', 1e-8)

        SSt = self.SS[:, self.SS[1, :] >= min(t_0 + self.N, self.time_limit)]

        x = model.addMVar((self.n, self.N + 1), lb = -1000, name = 'x')
        u = model.addMVar((self.m, self.N), lb = -1000, name = 'u')
        delta = model.addMVar(SSt.shape[1], vtype = GRB.BINARY, name = 'delta')
        B1 = model.addMVar(self.N, vtype = GRB.BINARY, name = 'B1')
        B2 = model.addMVar(self.N, vtype = GRB.BINARY, name = 'B2')

        model.addConstr(x[:,0] == x_0) # initial condition
        model.addConstr(x[:,self.N] == SSt[2:,:]@delta) # terminal constraint
        model.addConstr(delta.sum() == 1)
        terminal_cost = SSt[0,:]@delta
        stage_costs = 0

        for k in range(self.N):
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

            model.addConstr(x[0,k] <= -self.L + 400*B1[k])
            model.addConstr(x[0,k] >= self.L - 200*B2[k])
            model.addQConstr(x[1,k] + 20*(1 - B1[k]@B2[k]) >= 0.5*self.w_L + 0.5*self.w_C) # box constraint

            if t_0 + k >= self.time_limit:
                model.addConstr(x[0,k] >= self.xF[0])
                model.addConstr(x[1,k] == self.xF[1])
                model.addConstr(x[2,k] == self.xF[2])
                model.addConstr(x[3,k] == self.xF[3])

            stage_costs += u[:,k]@self.R@u[:,k]

        model.setObjective(terminal_cost + stage_costs, GRB.MINIMIZE)
        model.optimize()
        self.current_comp_time = model.Runtime

        self.x_star = x.x
        self.u_star = u.x

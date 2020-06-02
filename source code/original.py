from lmpc import *

class Original(LMPC):
    def __init__(self, T, dt, N, J, R):
        super().__init__(T, dt, N, J, R)
        self.name = 'original'

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
        model.setParam('NonConvex', 2)
        model.setParam('IntFeasTol', 1e-8)
        #model.setParam('FeasibilityTol', 1e-8)

        x = model.addMVar((self.n, self.N + 1), lb = -1000, name = 'x')
        u = model.addMVar((self.m, self.N), lb = -1000, name = 'u')
        T = model.addMVar(self.N, vtype = GRB.BINARY, name = 'T')
        delta = model.addMVar(self.SS.shape[1], vtype = GRB.BINARY, name = 'delta')
        B1 = model.addMVar(self.N, vtype = GRB.BINARY, name = 'B1')
        B2 = model.addMVar(self.N, vtype = GRB.BINARY, name = 'B2')

        model.addConstr(x[:,0] == x_0) # initial condition
        model.addConstr(x[:,self.N] == self.SS[1:,:]@delta) # terminal constraint
        model.addConstr(delta.sum() == 1)
        terminal_cost = self.SS[0,:]@delta
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

            model.addConstr(200 * T[k] >= self.xF[0] - x[0,k]) # far enough?
            model.addConstr(100 * T[k] >= x[1,k]@x[1,k] - 2*self.xF[1]*x[1,k] + self.xF[1]*self.xF[1]) # central enough?
            model.addConstr(100 * T[k] >= x[2,k]@x[2,k] - 2*self.xF[2]*x[2,k] + self.xF[2]*self.xF[2]) # fast enough?
            model.addConstr(100 * T[k] >= x[3,k]@x[3,k] - 2*self.xF[3]*x[3,k] + self.xF[3]*self.xF[3]) # steady enough?

            model.addConstr(x[0,k] <= -self.L + 400*B1[k])
            model.addConstr(x[0,k] >= self.L - 200*B2[k])
            model.addQConstr(x[1,k] + 20*(1 - B1[k]@B2[k]) >= 0.5*self.w_L + 0.5*self.w_C) # box constraint

            stage_costs += self.dt*T[k] + u[:,k]@self.R@u[:,k]

        model.setObjective(terminal_cost + stage_costs, GRB.MINIMIZE)
        model.optimize()
        self.current_comp_time = model.Runtime

        self.x_star = x.x
        self.u_star = u.x

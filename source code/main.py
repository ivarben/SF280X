from original import *
from original_limited import *
from spatial import *
from relax import *
from relax_limited import *

def main():
    T = 200 # 50s
    dt = 0.5
    J = 9 # Number of iterations of LMPC
    #N = 5 # control/ prediction horizon
    problems = []

    for r in [0.01, 100]:
        for N in [10, 20, 30, 40, 50]:
            print('####### r = ' + str(r) + ', N = ' + str(N) + ' #######')
            R = r*np.eye(2) # punishment on input

            problem = Original(T, dt, N, J, R) # LMPC instance
            #problem = OriginalLimited(T, dt, N, J, R)
            #problem = Spatial(T, dt, N, J, R)
            #problem = Relax(T, dt, N, J, R)
            #problem = RelaxLimited(T, dt, N, J, R)

            problem.solve() # Solve the LMPC
            problem.plot() # plot the results/ save the plots
            #problem.animate()
            problem.save()
            print(problem.iteration_costs)
            problems.append(problem)

    if problems[0].name == 'original' or problems[0].name == 'relax':
        plot_graphs(problems)
    else:
        plot_graphs_limited(problems)

def plot_graphs(problems):
    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    vehicle_L = Rectangle((-problems[0].l_C/2, -problems[0].w_C/2), problems[0].l_C, problems[0].w_C, label = 'Vehicle L', color = 'r')
    ax1.add_artist(vehicle_L)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax2.set_xlabel('time step')
    ax2.set_ylabel('x dot (m/s)')
    ax3.set_xlabel('time step')
    ax3.set_ylabel('y dot (m/s)')

    # road sides
    ax1.axhline(-0.5*problems[0].w_L, lw = 2, color = '0.5')
    ax1.axhline(1.5*problems[0].w_L, lw = 2, color = '0.5')
    ax1.axhline(0.5*problems[0].w_L, ls = '--', lw = 2, color = '0.7')

    if problems[0].name == 'relax' or problems[0].name == 'relax_limited':
        ax1.plot([-problems[0].d, problems[0].d], [-0.5*problems[0].w_L + 0.5*problems[0].w_C] * 2, lw = 1, color = 'r')
        ax1.plot([-problems[0].d, problems[0].d], [1.5*problems[0].w_L - 0.5*problems[0].w_C] * 2, lw = 1, color = 'r')
        ax1.plot([(-1.5*problems[0].w_L+problems[0].W)*problems[0].L/problems[0].W, (0.5*problems[0].w_L+problems[0].W)*problems[0].L/problems[0].W], [1.5*problems[0].w_L, -0.5*problems[0].w_L], lw = 1, ls = '--', color = '0.5')
        ax1.plot([(1.5*problems[0].w_L-problems[0].W)*problems[0].L/problems[0].W, -(0.5*problems[0].w_L+problems[0].W)*problems[0].L/problems[0].W], [1.5*problems[0].w_L, -0.5*problems[0].w_L], lw = 1, ls = '--', color = '0.5')
    else:
        ax1.plot([-problems[0].d, -problems[0].L], [-0.5*problems[0].w_L + 0.5*problems[0].w_C] * 2, lw = 1, color = 'r')
        ax1.plot([-problems[0].L, problems[0].L], [0.5*problems[0].w_L + 0.5*problems[0].w_C] * 2, lw = 1, color = 'r')
        ax1.plot([problems[0].L, problems[0].d], [-0.5*problems[0].w_L + 0.5*problems[0].w_C] * 2, lw = 1, color = 'r')
        ax1.plot([-problems[0].d, problems[0].d], [1.5*problems[0].w_L - 0.5*problems[0].w_C] * 2, lw = 1, color = 'r')

    for problem in problems:
        ax1.plot(problem.x_trajectory[0,:problem.completion[problem.J]+1,problem.J], problem.x_trajectory[1,:problem.completion[problem.J]+1,problem.J], marker = '.' , markersize = 4, lw = 0.75, label = 'N = ' + str(problem.N) + ', r = ' + str(problem.r))
        ax2.plot(problem.time[:problem.completion[problem.J]+1], problem.x_trajectory[2,:problem.completion[problem.J]+1,problem.J], lw = 0.75, label =  'N = ' + str(problem.N) + ', r = ' + str(problem.r))
        ax3.plot(problem.time[:problem.completion[problem.J]+1], problem.x_trajectory[3,:problem.completion[problem.J]+1,problem.J], lw = 0.75, label = 'N = ' + str(problem.N) + ', r = ' + str(problem.r))

    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'center right', borderaxespad=0.1)
    fig.tight_layout()
    plt.subplots_adjust(right=0.82)
    fig.set_size_inches(11, 7)
    fig.savefig('plots/' + 'TRAJ_SUMMARY_' + problems[0].name + '.pdf', bbox_inches='tight', dpi = 500)
    plt.close()

    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
    #ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1)
    ax3 = plt.subplot2grid((1, 2), (0, 1), colspan=1)
    ax1.set_title('r = 0.01')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    #ax2.set_title('r = 1')
    #ax2.set_xlabel('Iteration')
    #ax2.set_ylabel('Cost')
    ax3.set_title('r = 100')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Cost')

    for problem in problems:
        if problem.r == 0.01:
            ax1.plot(np.arange(problem.J+1), problem.iteration_costs, label = 'N= ' + str(problem.N))
        elif problem.r == 1:
            ax2.plot(np.arange(problem.J+1), problem.iteration_costs, label = 'N= ' + str(problem.N))
        elif problem.r == 100:
            ax3.plot(np.arange(problem.J+1), problem.iteration_costs, label = 'N= ' + str(problem.N))

    ax1.set_ylim(bottom = 0)
    #ax2.set_ylim(bottom = 0)
    ax3.set_ylim(bottom = 0)
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'center right', borderaxespad=0.1)
    fig.tight_layout()
    plt.subplots_adjust(right=0.92)
    fig.set_size_inches(11, 7)
    fig.savefig('plots/' + 'COST_SUMMARY_' + problems[0].name + '.pdf', bbox_inches='tight', dpi = 500)
    plt.close()

    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1)
    #ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=1)
    ax3 = plt.subplot2grid((1, 2), (0, 1), colspan=1)
    ax1.set_title('r = 0.01')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Computational Time (s)')
    #ax2.set_title('r = 1')
    #ax2.set_xlabel('Iteration')
    #ax2.set_ylabel('Mean Computational Time (s)')
    ax3.set_title('r = 100')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Mean Computational Time (s)')

    for problem in problems:
        if problem.r == 0.01:
            ax1.plot(np.arange(1,problem.J+1), problem.avgcomptime, label = 'N= ' + str(problem.N))
        elif problem.r == 1:
            ax2.plot(np.arange(1,problem.J+1), problem.avgcomptime, label = 'N= ' + str(problem.N))
        elif problem.r == 100:
            ax3.plot(np.arange(1,problem.J+1), problem.avgcomptime, label = 'N= ' + str(problem.N))

    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'center right', borderaxespad=0.1)
    fig.tight_layout()
    plt.subplots_adjust(right=0.92)
    fig.set_size_inches(11, 7)
    fig.savefig('plots/' + 'COMPT_SUMMARY_' + problems[0].name + '.pdf', bbox_inches='tight', dpi = 500)
    plt.close()

def plot_graphs_limited(problems):
    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    vehicle_L = Rectangle((-problems[0].l_C/2, -problems[0].w_C/2), problems[0].l_C, problems[0].w_C, label = 'Vehicle L', color = 'r')
    ax1.add_artist(vehicle_L)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax2.set_xlabel('time step')
    ax2.set_ylabel('x dot (m/s)')
    ax3.set_xlabel('time step')
    ax3.set_ylabel('y dot (m/s)')

    # road sides
    ax1.axhline(-0.5*problems[0].w_L, lw = 2, color = '0.5')
    ax1.axhline(1.5*problems[0].w_L, lw = 2, color = '0.5')
    ax1.axhline(0.5*problems[0].w_L, ls = '--', lw = 2, color = '0.7')

    if problems[0].name == 'relax' or problems[0].name == 'relax_limited':
        ax1.plot([-problems[0].d, problems[0].d], [-0.5*problems[0].w_L + 0.5*problems[0].w_C] * 2, lw = 1, color = 'r')
        ax1.plot([-problems[0].d, problems[0].d], [1.5*problems[0].w_L - 0.5*problems[0].w_C] * 2, lw = 1, color = 'r')
        ax1.plot([(-1.5*problems[0].w_L+problems[0].W)*problems[0].L/problems[0].W, (0.5*problems[0].w_L+problems[0].W)*problems[0].L/problems[0].W], [1.5*problems[0].w_L, -0.5*problems[0].w_L], lw = 1, ls = '--', color = '0.5')
        ax1.plot([(1.5*problems[0].w_L-problems[0].W)*problems[0].L/problems[0].W, -(0.5*problems[0].w_L+problems[0].W)*problems[0].L/problems[0].W], [1.5*problems[0].w_L, -0.5*problems[0].w_L], lw = 1, ls = '--', color = '0.5')
    else:
        ax1.plot([-problems[0].d, -problems[0].L], [-0.5*problems[0].w_L + 0.5*problems[0].w_C] * 2, lw = 1, color = 'r')
        ax1.plot([-problems[0].L, problems[0].L], [0.5*problems[0].w_L + 0.5*problems[0].w_C] * 2, lw = 1, color = 'r')
        ax1.plot([problems[0].L, problems[0].d], [-0.5*problems[0].w_L + 0.5*problems[0].w_C] * 2, lw = 1, color = 'r')
        ax1.plot([-problems[0].d, problems[0].d], [1.5*problems[0].w_L - 0.5*problems[0].w_C] * 2, lw = 1, color = 'r')

    for problem in problems:
        ax1.plot(problem.x_trajectory[0,:problem.completion[problem.J]+1,problem.J], problem.x_trajectory[1,:problem.completion[problem.J]+1,problem.J], marker = '.' , markersize = 4, lw = 0.75, label = 'N = ' + str(problem.N))
        ax2.plot(problem.time[:problem.completion[problem.J]+1], problem.x_trajectory[2,:problem.completion[problem.J]+1,problem.J], lw = 0.75, label =  'N = ' + str(problem.N))
        ax3.plot(problem.time[:problem.completion[problem.J]+1], problem.x_trajectory[3,:problem.completion[problem.J]+1,problem.J], lw = 0.75, label = 'N = ' + str(problem.N))

    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, loc = 'center right', borderaxespad=0.1)
    fig.tight_layout()
    plt.subplots_adjust(right=0.82)
    fig.set_size_inches(11, 7)
    fig.savefig('plots/' + 'TRAJ_SUMMARY_' + problems[0].name + '.pdf', bbox_inches='tight', dpi = 500)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')

    for problem in problems:
        ax.plot(np.arange(problem.J+1), problem.iteration_costs, label = 'N= ' + str(problem.N))

    ax.set_ylim(bottom = 0)
    fig.legend(loc = 'center right')
    fig.set_size_inches(11, 7)
    fig.savefig('plots/' + 'COST_SUMMARY_' + problems[0].name + '.pdf', bbox_inches='tight', dpi = 500)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Computational Time (s)')

    for problem in problems:
        ax.plot(np.arange(1,problem.J+1), problem.avgcomptime, label = 'N= ' + str(problem.N))

    fig.legend(loc = 'center right')
    fig.set_size_inches(11, 7)
    fig.savefig('plots/' + 'COMPT_SUMMARY_' + problems[0].name + '.pdf', bbox_inches='tight', dpi = 500)
    plt.close()

if __name__ == '__main__':
    main()

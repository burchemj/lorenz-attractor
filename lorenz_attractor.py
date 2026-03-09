import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# note that sigma=10, beta=8/3, rho=28 are the values for the lorenz attractor
def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    dxdt = sigma * (y-x)
    dydt = x * (rho - z) - y
    dzdt = (x*y) - (beta*z)
    return [dxdt, dydt, dzdt]

t_span = (0,40)
t_eval = np.linspace(*t_span, 5000)
initial_conditions = [[5., 5., 5.], [5.1, 5.1, 5.2], [4.8, 5, 4.8]]

def integrate_lorenz(initial_state, t_span, t_eval):
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')
    return sol.y

trajectories = [integrate_lorenz(init, t_span, t_eval) for init in initial_conditions]

traj1, traj2 = trajectories[0], trajectories[1]
distance = np.sqrt((traj1[0]-traj2[0])**2 + (traj1[1]-traj2[1])**2 + (traj1[2]-traj2[2])**2)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')

ax.set_xlim([-25, 25])
ax.set_ylim([-35, 35])
ax.set_zlim([0, 50])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Lorenz Attractor')

ax2 = fig.add_subplot(122)
div_line, = ax2.plot([],[], color='blue')
ax2.set_xlim([0,t_span[1]])
ax2.set_ylim([0,max(distance)*1.1])
ax2.set_xlabel('time')
ax2.set_ylabel('distance')
ax2.set_title('Divergence of trajectories')


colors = ['red', 'green', 'blue']
lines = [ax.plot([], [], [], lw=1, color=colors[i], markersize=0.5)[0] for i in range(3)]

def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    div_line.set_data([],[])
    return lines + [div_line]

def update(frame):
    for i, traj in enumerate(trajectories):
        x, y, z = traj
        lines[i].set_data(x[:frame], y[:frame])
        lines[i].set_3d_properties(z[:frame])
    
    div_line.set_data(t_eval[:frame], distance[:frame])
    
    return lines + [div_line]

frames = len(t_eval)
anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval = 30, blit=False)

plt.show()
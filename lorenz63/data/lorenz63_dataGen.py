""" 
Generate trajectory data for the Lorenz '63 system.
"""

# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

    
# Parameters
t_span = (0, 100)
dt = 0.01
T = np.arange(t_span[0], t_span[1], dt)


num_initial_states = 1000
initial_states = np.random.uniform(low=-1, high=1, size=(num_initial_states, 3))
trajectories = []

for initial_state in initial_states:
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=T, method='RK45')
    trajectory = sol.y.T  # shape (num_time_steps, 3)
    discard_fraction = 0.5  # Discard the first 50% of the trajectory to remove transients
    discard_index = int(len(trajectory) * discard_fraction)
    trajectory = trajectory[discard_index:]
    trajectories.append(trajectory)

trajectories = np.array(trajectories)  # (num_initial_states, num_time_steps_after_discard, 3)

# Optional: Save or plot
save = True
if save:
    np.save('lorenz63/data/lorenz63X.npy', trajectories)             

plot = False
if plot:
    trajectories_ind = [0]
    # Plot trajectory
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    ax.plot(trajectories[trajectories_ind, :, 0], trajectories[trajectories_ind, :, 1], trajectories[trajectories_ind, :, 2])
    ax.set_title("Lorenz Attractor")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()





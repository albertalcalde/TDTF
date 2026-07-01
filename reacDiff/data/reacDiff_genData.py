import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Lx = 1
Nx = 512
D = 0.0322307
eps = 0.01 
alpha = 0.01
dealias = 3/2
stop_sim_time = 100
timestepper = d3.RK222
timestep = 1e-4
dtype = np.float64

# Bases
xcoord = d3.Coordinate('x')
dist = d3.Distributor(xcoord, dtype=dtype)
xbasis = d3.Chebyshev(xcoord, size = Nx, bounds=(0, Lx), dealias=dealias)

# Fields
u = dist.Field(name='u', bases=xbasis)
v = dist.Field(name='v', bases=xbasis)
tau_u1 = dist.Field(name='tau_u1')
tau_u2 = dist.Field(name='tau_u2')
tau_v1 = dist.Field(name='tau_v1')
tau_v2 = dist.Field(name='tau_v2')

# Substitutions
dx = lambda A: d3.Differentiate(A, xcoord)

# Tau polynomials
tau_basis = xbasis.derivative_basis(2)
p1 = dist.Field(bases=tau_basis)
p2 = dist.Field(bases=tau_basis)
p1['c'][-1] = 1
p2['c'][-2] = 2


# Problem
problem = d3.IVP([u, v, tau_u1, tau_u2, tau_v1, tau_v2], namespace=locals())

problem.add_equation("dt(u) - D*dx(dx(u)) + tau_u1*p1 + tau_u2*p2 = (v - u*u - u*u*u)/eps")
problem.add_equation("dt(v) - D*dx(dx(v)) + tau_v1*p1 + tau_v2*p2 = -u + alpha")

# Add boundary conditions
problem.add_equation("u(x='left') = -2")
problem.add_equation("u(x='right') = -2")
problem.add_equation("v(x='left') = -4")
problem.add_equation("v(x='right') = -4")

# Initial conditions
x = dist.local_grid(xbasis)

# Initial conditions
u['g'] = -2  
v['g'] = -4  

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Main loop
tmin_plot = 0

u_list = [u['g',1].copy()]
v_list = [v['g',1].copy()]
t_list = [solver.sim_time]
while solver.proceed:
    solver.step(timestep)
    if solver.iteration % 100 == 0:
        logger.info('Iteration=%i, Time=%e, dt=%e, norm=%e' %(solver.iteration, solver.sim_time, timestep, np.linalg.norm(u['g'])))
    if solver.sim_time >= tmin_plot and solver.iteration % 25 == 0:
        u_list.append(u['g',1].copy())
        v_list.append(v['g',1].copy())
        t_list.append(solver.sim_time)

save = True
if save:
    np.savez("./stableODE/reacDiffU_D=%.8f.npz" % D, x = x.copy(), t = np.array(t_list), u = np.array(u_list), v = np.array(v_list))
else:
    arrays = np.load(f"./stableODE/reacDiffU_D={D:.8f}.npz")
    x = arrays['x']    # shape (Nx,)
    t = arrays['t']    # shape (K,)
    u = arrays['u'].T  # shape (Nx, K)

# remove transient
transient_time = 15.0
transient_index = np.searchsorted(t, transient_time)
u = u[:, transient_index:]
t = t[transient_index:]

# Center the data by subtracting the mean
u_mean = np.mean(u, axis=1, keepdims=True)
centered_u = u - u_mean

U, S, Vh = np.linalg.svd(centered_u, full_matrices=False, compute_uv=True)

pod_modes = 3
Phi = U[:, :pod_modes]  # POD modes

pod_coeffs = np.dot(Phi.T, centered_u)  # POD coefficients

# Save POD coefficients
save_pod = True
if save_pod:
    np.savez(f"./stableODE/PODcoeffs_d={pod_modes}_D={D:.8f}.npz", t=t.copy(), pod_coeffs=pod_coeffs.copy(), pod_modes=pod_modes, Phi=Phi.copy(), u_mean=u_mean.copy())

""" 
Generate data from a sinusoidal signal:
x_t = sin(t h) for some h > 0.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

    
# Parameters
Tf = 4 * 2 * np.pi          # save 4 periods
N_t = 200                   # Points number of the temporal domain
h = Tf / N_t                 # Time step
T = np.linspace(0,Tf,N_t)   # Temporal array

X = np.sin(T)                # sinusoidal signal

save = True
if save:
    np.save('sinus/data/sinusX.npy', X)
    
# %% Plotting
plt.plot(T, X)
plt.xlabel(r'$t$')
plt.ylabel(r'$\sin(t)$')
plt.title('Sinusoidal signal')
plt.grid(True)
plt.show()

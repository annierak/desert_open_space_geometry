import time
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib
import matplotlib.patches
matplotlib.use("Agg")
import sys
import itertools
import h5py
import json
import cPickle as pickle
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np

import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.wind_models as wind_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
from pompy import models,processors

random_state = np.random.RandomState(1)

dt = 0.25
simulation_time = 50.*60.
wind_angle = 0.
wind_mag = 1.
wind_param = {
            'speed': wind_mag,
            'angle': wind_angle,
            'evolving': False,
            'wind_dt': None,
            'dt': dt
            }
wind_field_noiseless = wind_models.WindField(param=wind_param)

#traps
source_locations = [(0.,0.),]
source_pos = scipy.array([scipy.array(tup) for tup in source_locations]).T


trap_param = {
'source_locations' : [source_pos],
'source_strengths' : [1.],
'epsilon'          : 0.01,
'trap_radius'      : 1.,
'source_radius'    : 1000.
}

traps = trap_models.TrapModel(trap_param)

#Odor arena
xlim = (0., 1200.)
ylim = (-50., 50.)
sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])

wind_region = models.Rectangle(xlim[0]*2,ylim[0]*2,
xlim[1]*2,ylim[1]*2)
im_extents = xlim[0], xlim[1], ylim[0], ylim[1]

#lazy plume parameters
puff_mol_amount = 1.
r_sq_max=20;epsilon=0.00001;N=1e6

lazyPompyPlumes = models.OnlinePlume(sim_region, source_pos, wind_field_noiseless,
    simulation_time,dt,r_sq_max,epsilon,puff_mol_amount,N)

#Directly inquire odor values downwind of the plume
n_samples = 1000
target_locations_x = np.linspace(1,500,n_samples)
target_locations_y = np.zeros_like(target_locations_x)


plt.figure()

for j in range(100):
    conc_values = lazyPompyPlumes.value(target_locations_x,target_locations_y)
    plt.plot(target_locations_x,conc_values)
    plt.ylim([0,3.])
    # raw_input()
    plt.show()

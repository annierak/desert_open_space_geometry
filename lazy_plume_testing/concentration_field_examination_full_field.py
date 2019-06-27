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
number_sources = 8
radius_sources = 1000.0
trap_radius = 0.5
location_list, strength_list = utility.create_circle_of_sources(number_sources,
                radius_sources,None)
trap_param = {
        'source_locations' : location_list,
        'source_strengths' : strength_list,
        'epsilon'          : 0.01,
        'trap_radius'      : trap_radius,
        'source_radius'    : radius_sources
}

traps = trap_models.TrapModel(trap_param)

#Odor arena
xlim = (-1500., 1500.)
ylim = (-1500., 1500.)
sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])
wind_region = models.Rectangle(xlim[0]*2,ylim[0]*2,
xlim[1]*2,ylim[1]*2)
im_extents = xlim[0], xlim[1], ylim[0], ylim[1]

source_pos = scipy.array([scipy.array(tup) for tup in traps.param['source_locations']]).T

#lazy plume parameters
puff_mol_amount = 1.
r_sq_max=20;epsilon=0.00001;N=1e6

lazyPompyPlumes = models.OnlinePlume(sim_region, source_pos, wind_field_noiseless,
    simulation_time,dt,r_sq_max,epsilon,puff_mol_amount,N)



plt.figure()

conc_im = lazyPompyPlumes.conc_im(im_extents,samples=150)

plt.imshow(conc_im,extent=im_extents,aspect='auto')

plt.show()

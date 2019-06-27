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
wind_angle = np.pi/4
wind_mag = 1.
wind_param = {
            'speed': wind_mag,
            'angle': wind_angle,
            'evolving': False,
            'wind_dt': None,
            'dt': dt
            }
wind_field_noiseless = wind_models.WindField(param=wind_param)

#sources
location_list = [(0,10),(0,20),(0,-10)]
source_pos = scipy.array([scipy.array(tup) for tup in location_list]).T

#Odor arena
xlim = (-10., 200.)
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



plt.figure()

conc_im = lazyPompyPlumes.conc_im(im_extents,samples=150)

plt.imshow(conc_im,extent=im_extents,aspect='equal',origin='lower',cmap='bone_r')


plt.figure()

#For a point of comparison, use the Gaussian plumes to show what the image should look like approximately
gaussianfitPlumes = models.GaussianFitPlume(source_pos.T,wind_angle,wind_mag)
conc_im = gaussianfitPlumes.conc_im(im_extents,samples=150)
plt.imshow(conc_im,extent=im_extents,aspect='equal',origin='lower',cmap='bone_r')


plt.show()

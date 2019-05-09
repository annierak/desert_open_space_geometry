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


wind_angle = 0.
wind_mag = 1.

dt = 0.25
plume_dt = 0.25
times_real_time = 60.
frame_rate = 20
capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))

simulation_time = 50.*60. #seconds
release_delay = 30.*60#/(wind_mag)

t_start = 0.0
t = 0. - release_delay

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

#Wind and plume objects

#Odor arena
xlim = (0., 1200.)
ylim = (-500., 500.)
sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])

wind_region = models.Rectangle(xlim[0]*2,ylim[0]*2,
xlim[1]*2,ylim[1]*2)
im_extents = xlim[0], xlim[1], ylim[0], ylim[1]

#wind model setup
diff_eq = False
constant_wind_angle = wind_angle
aspect_ratio= (xlim[1]-xlim[0])/(ylim[1]-ylim[0])
noise_gain=3.
noise_damp=0.071
noise_bandwidth=0.71
wind_grid_density = 200
Kx = Ky = 10000 #highest value observed to not cause explosion: 10000
wind_field = models.WindModel(wind_region,int(wind_grid_density*aspect_ratio),
wind_grid_density,noise_gain=noise_gain,noise_damp=noise_damp,
noise_bandwidth=noise_bandwidth,Kx=Kx,Ky=Ky,
diff_eq=diff_eq,angle=constant_wind_angle,mag=wind_mag)


# Set up plume model
centre_rel_diff_scale = 2.
puff_release_rate = 10
puff_spread_rate=0.005
puff_init_rad = 0.01
max_num_puffs=int(2e5)

plume_model = models.PlumeModelCopy(
    sim_region, source_pos, wind_field,simulation_time+release_delay,
    plume_dt,plume_cutoff_radius=1500,
    centre_rel_diff_scale=centre_rel_diff_scale,
    puff_release_rate=puff_release_rate,
    puff_init_rad=puff_init_rad,puff_spread_rate=puff_spread_rate,
    max_num_puffs=max_num_puffs)

#lazy plume parameters
puff_mol_amount = 1.
r_sq_max=20;epsilon=0.00001;N=1e6

wind_param = {
            'speed': wind_mag,
            'angle': wind_angle,
            'evolving': False,
            'wind_dt': None,
            'dt': dt
            }
wind_field_noiseless = wind_models.WindField(param=wind_param)



lazyPompyPlumes = models.OnlinePlume(sim_region, source_pos, wind_field_noiseless,
    simulation_time,dt,r_sq_max,epsilon,puff_mol_amount,N)


while t<simulation_time:
    for k in range(capture_interval):
        #update flies
        print('t: {0:1.2f}'.format(t))
        #update the swarm
        for j in range(int(dt/plume_dt)):
            wind_field.update(plume_dt)
            plume_model.update(plume_dt,t,verbose=True)
        if t>0.:
            # values = lazyPompyPlumes.value(np.zeros(10),100.*np.ones(10))
            # raw_input()
            plt.figure()

            # plt.hist(plume_model.puffs[:,:,3][~np.isnan(plume_model.puffs[:,:,3])].flatten()[::10])

            plt.scatter(plume_model.puffs[:,:,0].flatten()[::],
                plume_model.puffs[:,:,1].flatten()[::],alpha=0.1,
                c=plume_model.puffs[:,:,3].flatten()[::]) #color by r_sq
            plt.xlim((0., 1800.))
            plt.ylim((-30., 30.))
            plt.title(str(np.sum(~np.isnan(plume_model.puffs[:,:,0].flatten()))))
            plt.colorbar()

            plt.show()
        t+= dt

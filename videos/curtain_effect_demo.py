import time
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib
import matplotlib.patches
matplotlib.use("Agg")
import sys
import itertools
import h5py
import json
import cPickle as picklek
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np

import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.wind_models as wind_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt
from pompy import models,processors


file_name = 'curtain_effect_demo_frozen_plumes'
output_file = file_name+'.pkl'

detection_thresholds = [0.05,0.1,0.2]

dt = 0.25
plume_dt = 0.25
frame_rate = 20
times_real_time = 20 # seconds of simulation / sec in video
capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))

simulation_time = 12.*60. #seconds
release_delay = 25.*60#/(wind_mag)
# release_delay = 10.*60#/(wind_mag)

t_start = 0.0
t = 0. - release_delay



# Set up figure
fig = plt.figure(figsize=(15, 5))

# #Video
FFMpegWriter = animate.writers['ffmpeg']
metadata = {'title':file_name,}
writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
writer.setup(fig, file_name+'.mp4', 500)

wind_angle = 7*scipy.pi/8.
wind_mag = 1.4

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

#Wind and plume objects

source_pos = scipy.array([scipy.array(tup) for tup in traps.param['source_locations']])

#Odor arena
xlim = (-2500., 1500.)
ylim = (-1500., 2500.)
sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])
wind_region = models.Rectangle(xlim[0]*1.2,ylim[0]*1.2,
xlim[1]*1.2,ylim[1]*1.2)

source_pos = scipy.array([scipy.array(tup) for tup in traps.param['source_locations']]).T

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
puff_mol_amount = 1.
centre_rel_diff_scale = 2.
# puff_release_rate = 0.001
puff_release_rate = 10
puff_spread_rate=0.005
puff_init_rad = 0.01
max_num_puffs=int(2e5)
# max_num_puffs=100

plume_model = models.PlumeModel(
    sim_region, source_pos, wind_field,simulation_time+release_delay,
    plume_dt,
    centre_rel_diff_scale=centre_rel_diff_scale,
    puff_release_rate=puff_release_rate,
    puff_init_rad=puff_init_rad,puff_spread_rate=puff_spread_rate,
    max_num_puffs=max_num_puffs)

# Create a concentration array generator
array_z = 0.01

array_dim_x = 1000
array_dim_y = array_dim_x
puff_mol_amount = 1.
array_gen = processors.ConcentrationArrayGenerator(
    sim_region, array_z, array_dim_x, array_dim_y, puff_mol_amount)



#Setup fly swarm
wind_slippage = (0.,1.)
swarm_size=2000
use_empirical_release_data = False

beta = 1.
release_times = scipy.random.exponential(beta,(swarm_size,))
kappa = 2.

heading_data=None

swarms = []

for detection_threshold in detection_thresholds:

    swarm_param = {
                'swarm_size'          : swarm_size,
                'heading_data'        : None,
                'initial_heading'     : scipy.radians(scipy.random.uniform(0.0,360.0,(swarm_size,))),
                'x_start_position'    : scipy.zeros(swarm_size),
                'y_start_position'    : scipy.zeros(swarm_size),
                'flight_speed'        : np.full((swarm_size,), 1.5),
                'release_time'        : release_times,
                'release_delay'       : release_delay,
                'cast_interval'       : [1.,3.],
                'wind_slippage'       : wind_slippage,
                'odor_thresholds'     : {
                    'lower': 0.0005,
                    'upper': detection_threshold
                    },
                'schmitt_trigger':False,
                'low_pass_filter_length':3., #seconds
                'dt_plot': capture_interval*dt,
                't_stop':simulation_time,
                'cast_timeout': 20.,
                'surging_error_std'   : scipy.radians(1e-10),
                'airspeed_saturation':False
                }



    swarm = swarm_models.BasicSwarmOfFlies(wind_field_noiseless,traps,param=swarm_param,
        start_type='fh',track_plume_bouts=False,track_arena_exits=False)

    swarms.append(swarm)



#Conc array gen to be used for the flies
sim_region_tuple = plume_model.sim_region.as_tuple()
box_min,box_max = sim_region_tuple[1],sim_region_tuple[2]

r_sq_max=20;epsilon=0.00001;N=1e6

array_gen_flies = processors.ConcentrationValueFastCalculator(
            box_min,box_max,r_sq_max,epsilon,puff_mol_amount,N)


#Initial concentration plotting

conc_array = array_gen.generate_single_array(plume_model.puffs)
xmin = sim_region.x_min; xmax = sim_region.x_max
ymin = sim_region.y_min; ymax = sim_region.y_max
im_extents = (xmin,xmax,ymin,ymax)

axes = [];conc_ims = []

for i,detection_threshold in enumerate(detection_thresholds):
    ax = fig.add_subplot(1,3,i+1);axes.append(ax)
    # vmin,vmax = 0.,detection_threshold
    cmap = matplotlib.colors.ListedColormap(['white', 'grey'])
    conc_im = ax.imshow(conc_array.T[::-1]>detection_threshold, extent=im_extents,
    vmin=0., vmax=1.,
    cmap=cmap)#,interpolation='bilinear')
    conc_ims.append(conc_im)

    xmin,xmax,ymin,ymax = -1500,1200,-1200,1500

    buffr = 100
    ax.set_xlim((xmin-buffr,xmax+buffr))
    ax.set_ylim((ymin-buffr,ymax+buffr))

#Initial fly plotting
#Sub-dictionary for color codes for the fly modes
Mode_StartMode = 0
Mode_FlyUpWind = 1
Mode_CastForOdor = 2
Mode_Trapped = 3


edgecolor_dict = {Mode_StartMode : 'blue',
Mode_FlyUpWind : 'red',
Mode_CastForOdor : 'red',
Mode_Trapped :   'black'}

facecolor_dict = {Mode_StartMode : 'blue',
Mode_FlyUpWind : 'red',
Mode_CastForOdor : 'white',
Mode_Trapped :   'black'}


fly_edgecolors = [edgecolor_dict[mode] for mode in swarm.mode]
fly_facecolors = [facecolor_dict[mode] for mode in swarm.mode]
fly_dotss = []
for ax in axes:
    fly_dots = ax.scatter(swarm.x_position, swarm.y_position,
        edgecolor=fly_edgecolors,facecolor = fly_facecolors,alpha=0.9)
    fly_dotss.append(fly_dots)

    (xmin,xmax) = ax.get_xlim();(ymin,ymax) = ax.get_ylim()



    #traps
    for x,y in traps.param['source_locations']:

        #Black x
        ax.scatter(x,y,marker='x',s=50,c='k')

    # ax.axis('off')

    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    labelbottom=False,
    labelleft=False)



fig.patch.set_facecolor('white')

plt.ion()
# plt.show()
# raw_input()
while t<simulation_time:
    for k in range(capture_interval):
        #update flies
        print('t: {0:1.2f}'.format(t))
        for j in range(int(dt/plume_dt)):
            wind_field.update(plume_dt)
            plume_model.update(plume_dt,verbose=True)
        if t>0.:
            for swarm in swarms:
                swarm.update(t,dt,wind_field_noiseless,array_gen_flies,traps,plumes=plume_model,
                    pre_stored=False)
        t+= dt
    if t>0:
        for fly_dots,swarm in zip(fly_dotss,swarms):
            fly_dots.set_offsets(scipy.c_[swarm.x_position,swarm.y_position])

            fly_edgecolors = [edgecolor_dict[mode] for mode in swarm.mode]
            fly_facecolors =  [facecolor_dict[mode] for mode in swarm.mode]

            fly_dots.set_edgecolor(fly_edgecolors)
            fly_dots.set_facecolor(fly_facecolors)

        if t<2:

            conc_array = array_gen.generate_single_array(plume_model.puffs)



            for detection_threshold,conc_im in zip(detection_thresholds,conc_ims):
                conc_array_bool = 10.*(conc_array>detection_threshold).astype(int)
                blurred_conc_array = scipy.ndimage.gaussian_filter(conc_array_bool, sigma=3)
                conc_im.set_data(blurred_conc_array.T[::-1])

        # plt.pause(0.0001)
        writer.grab_frame()

        # raw_input()



writer.finish()

with open(output_file, 'w') as f:
    pickle.dump((wind_field,swarm),f)

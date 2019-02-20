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

num_iterations = 10

def f(wind_angle,i):

    wind_mag = 1.4

    file_name = 'trap_arrival_by_wind_live_coarse_dt'
    file_name = file_name +'_wind_mag_'+str(wind_mag)+'_wind_angle_'+str(wind_angle)[0:4]+'_iter_'+str(i)
    output_file = file_name+'.pkl'

    dt = 0.25
    plume_dt = 0.25
    frame_rate = 20
    times_real_time = 20 # seconds of simulation / sec in video
    capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))

    simulation_time = 50.*60. #seconds
    release_delay = 30.*60#/(wind_mag)

    t_start = 0.0
    t = 0. - release_delay


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

    #Odor arena
    xlim = (-1500., 1500.)
    ylim = (-1500., 1500.)
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
    centre_rel_diff_scale = 2.
    # puff_release_rate = 0.001
    puff_release_rate = 10
    puff_spread_rate=0.005
    puff_init_rad = 0.01
    max_num_puffs=int(2e5)
    # max_num_puffs=100

    plume_model = models.PlumeModel(
        sim_region, source_pos, wind_field,simulation_time+release_delay,
        plume_dt,plume_cutoff_radius=1500,
        centre_rel_diff_scale=centre_rel_diff_scale,
        puff_release_rate=puff_release_rate,
        puff_init_rad=puff_init_rad,puff_spread_rate=puff_spread_rate,
        max_num_puffs=max_num_puffs)


    puff_mol_amount = 1.

    #Setup fly swarm
    wind_slippage = (0.,1.)
    swarm_size=2000
    use_empirical_release_data = False

    #Grab wind info to determine heading mean
    wind_x,wind_y = wind_mag*scipy.cos(wind_angle),wind_mag*scipy.sin(wind_angle)

    beta = 1.
    release_times = scipy.random.exponential(beta,(swarm_size,))
    kappa = 2.

    heading_data=None

    swarm_param = {
            'swarm_size'          : swarm_size,
            'heading_data'        : heading_data,
            'initial_heading'     : scipy.radians(scipy.random.uniform(0.0,360.0,(swarm_size,))),
            'x_start_position'    : scipy.zeros(swarm_size),
            'y_start_position'    : scipy.zeros(swarm_size),
            'flight_speed'        : scipy.full((swarm_size,), 1.5),
            'release_time'        : release_times,
            'release_delay'       : release_delay,
            'cast_interval'       : [1, 3],
            'wind_slippage'       : wind_slippage,
            'odor_thresholds'     : {
                'lower': 0.0005,
                'upper': 0.05
                },
            'schmitt_trigger':False,
            'low_pass_filter_length':3, #seconds
            'dt_plot': capture_interval*dt,
            't_stop':3000.,
            'cast_timeout':20,
            'airspeed_saturation':True
            }

    swarm = swarm_models.BasicSwarmOfFlies(wind_field_noiseless,traps,param=swarm_param,
        start_type='fh',track_plume_bouts=False,track_arena_exits=False)

    # xmin,xmax,ymin,ymax = -1000,1000,-1000,1000


    #Conc array gen to be used for the flies
    sim_region_tuple = plume_model.sim_region.as_tuple()
    box_min,box_max = sim_region_tuple[1],sim_region_tuple[2]

    r_sq_max=20;epsilon=0.00001;N=1e6

    array_gen_flies = processors.ConcentrationValueFastCalculator(
                box_min,box_max,r_sq_max,epsilon,puff_mol_amount,N)

    while t<simulation_time:
        for k in range(capture_interval):
            #update flies
            print('t: {0:1.2f}'.format(t))
            #update the swarm
            for j in range(int(dt/plume_dt)):
                wind_field.update(plume_dt)
                plume_model.update(plume_dt,verbose=True)
            if t>0.:
                swarm.update(t,dt,wind_field_noiseless,array_gen_flies,traps,plumes=plume_model,
                    pre_stored=False)
            t+= dt

    with open(output_file, 'w') as f:
        pickle.dump((wind_field_noiseless,swarm),f)


def g(wind_angle,num_iterations=num_iterations):

    wind_mag = 1.4

    #Loop through pickle files for that parameter value and merge counts
    for i in range(num_iterations):

        file_name = 'trap_arrival_by_wind_live_coarse_dt'
        file_name = file_name +'_wind_mag_'+str(wind_mag)+'_wind_angle_'+str(wind_angle)[0:4]+'_iter_'+str(i)
        output_file = file_name+'.pkl'

        with open(output_file, 'r') as f:
            (_,swarm) = pickle.load(f)

        if i==0:
            sim_trap_counts_cumul = np.zeros(swarm.num_traps)

        sim_trap_counts_cumul += swarm.get_trap_counts()

    #Trap arrival plot

    #Set 0s to 1 for plotting purposes
    sim_trap_counts_cumul[sim_trap_counts_cumul==0] = .5
    trap_locs = (2*scipy.pi/swarm.num_traps)*scipy.array(swarm.list_all_traps())
    radius_scale = 0.3
    plot_size = 1.5
    fig=plt.figure(200+int(10*wind_mag))
    ax = plt.subplot(aspect=1)
    trap_locs_2d = [(scipy.cos(trap_loc),scipy.sin(trap_loc)) for trap_loc in trap_locs]
    patches = [plt.Circle(
        center, size) for center, size in zip(
            trap_locs_2d, radius_scale*sim_trap_counts_cumul/max(sim_trap_counts_cumul))]
    biggest_trap_loc = trap_locs_2d[
        np.where(sim_trap_counts_cumul==max(sim_trap_counts_cumul))[0][0]]
    coll = matplotlib.collections.PatchCollection(patches, facecolors='blue',edgecolors='blue')
    ax.add_collection(coll)
    ax.set_ylim([-plot_size,plot_size]);ax.set_xlim([-plot_size,plot_size])
    ax.set_xticks([])
    ax.set_xticklabels('')
    ax.set_yticks([])
    ax.set_yticklabels('')
    #Label max trap count
    ax.text(biggest_trap_loc[0],biggest_trap_loc[1],
        str(max(sim_trap_counts_cumul)),horizontalalignment='center',
            verticalalignment='center',fontsize=18,color='white')
    #Wind arrow
    plt.arrow(0.5, 0.5, 0.1*scipy.cos(wind_angle), 0.1*scipy.sin(wind_angle),transform=ax.transAxes,color='b',
        width=0.001)
    # ax.text(0.55, 0.5,'Wind',transform=ax.transAxes,color='b')
    ax.text(0,1.5,'N',horizontalalignment='center',verticalalignment='center',fontsize=25)
    ax.text(0,-1.5,'S',horizontalalignment='center',verticalalignment='center',fontsize=25)
    ax.text(1.5,0,'E',horizontalalignment='center',verticalalignment='center',fontsize=25)
    ax.text(-1.5,0,'W',horizontalalignment='center',verticalalignment='center',fontsize=25)
    # plt.title('Simulated')
    fig.patch.set_facecolor('white')
    plt.axis('off')
    ax.text(0,1.7,'Trap Counts'+' (Wind Mag: '+str(wind_mag)[0:3]+')',horizontalalignment='center',verticalalignment='center',fontsize=20)

    file_name = 'trap_arrival_by_wind_live_coarse_dt'
    png_file_name = file_name +'_wind_mag_'+str(wind_mag)+'_wind_angle_'+str(wind_angle)[0:4]

    plt.savefig(png_file_name+'.png',format='png')

import multiprocessing
from itertools import product
from contextlib import contextmanager

def f_unpack(args):
    return f(*args)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

angles = list(np.linspace(3*np.pi/2,7*np.pi/4,6))
iterations = range(num_iterations)

# angles = [np.pi,np.pi/2]

with poolcontext(processes=6) as pool:
    pool.map(f_unpack, product(angles, iterations))

pool = multiprocessing.Pool(processes=6)

pool.map(g, angles)

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

num_iterations = 1

def f(wind_angle,i):

    random_state = np.random.RandomState(i)

    wind_mag = 1.2

    file_name = 'trap_time_course_by_wind'
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
    wind_region = models.Rectangle(xlim[0]*2,ylim[0]*2,
    xlim[1]*2,ylim[1]*2)

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
    noise_bandwidth=noise_bandwidth,Kx=Kx,Ky=Ky,noise_rand=random_state,
    diff_eq=diff_eq,angle=constant_wind_angle,mag=wind_mag)


    # Set up plume model
    plume_width_factor = 1.
    centre_rel_diff_scale = 2.*plume_width_factor
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
        max_num_puffs=max_num_puffs,prng=random_state)


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
    box_min,box_max = -3000.,3000.

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

    #wind_angle = 7*np.pi/4
    wind_mag = 1.2
    swarms = []

    #Loop through pickle files for that parameter value and merge counts
    for i in range(num_iterations):

        # file_name = 'trap_time_course_by_wind'
        file_name = 'trap_arrival_by_wind_live_coarse_dt'
        #file_name = 'trap_arrival_by_wind_fit_gauss'
        # file_name = 'trap_arrival_by_wind_adjusted_fit_gauss'
        # file_name = file_name +'_wind_mag_'+str(wind_mag)+'_wind_angle_'+str(wind_angle)[0:4]+'_iter_'+str(i)
        file_name = file_name +'_wind_mag_'+str(wind_mag)#+'_wind_angle_'+str(wind_angle)[0:4]+'_iter_'+str(i)


        output_file = file_name+'.pkl'

        with open(output_file, 'r') as f:
            # (_,swarm) = pickle.load(f)
            swarm = pickle.load(f)

        swarms.append(swarm)


    #Trap arrival plot

    num_bins = 120

    trap_num_list = swarms[0].get_trap_nums()


    peak_counts = scipy.zeros(len(trap_num_list))
    peak_counts = scipy.zeros(8)
    rasters = []

    fig = plt.figure(figsize=(7, 11))

    fig.patch.set_facecolor('white')

    labels = ['N','NE','E','SE','S','SW','W','NW']

    sim_reorder = scipy.array([3,2,1,8,7,6,5,4])

    #Simulated histogram
    # for i in range(len(trap_num_list)):
    for i in range(8):

        row = sim_reorder[i]-1
        # ax = plt.subplot2grid((len(trap_num_list),1),(i,0))
        ax = plt.subplot2grid((8,1),(row,0))
        t_sim = scipy.concatenate(tuple(swarm.get_time_trapped(i) for swarm in swarms))

        if len(t_sim)==0:
            ax.set_xticks([0,10,20,30,40,50])
            trap_total = 0
            pass
        else:
            t_sim = t_sim/60.
            (n, bins, patches) = ax.hist(t_sim,num_bins,cumulative=True,
            histtype='step',
            range=(0,max(t_sim)))

            # n = n/num_iterations
            # trap_total = int(sum(n))
            # trap_total = int(n[-1])
            try:
                peak_counts[i]=max(n)
            except(IndexError):
                peak_counts[i]=0

        if sim_reorder[i]-1==0:
            ax.set_title('Cumulative Trap Arrivals \n Narrow Plumes, Wind Mag: '+str(wind_mag)+', Wind Angle: '+str(wind_angle)[0:4])

        ax.set_xlim([0,50])
        # ax.set_yticks([ax.get_yticks()[0],ax.get_yticks()[-1]])
        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True)
        # ax.text(-0.1,0.5,str(trap_total),transform=ax.transAxes,fontsize=20,horizontalalignment='center')
        # ax.text(-0.01,1,trap_total,transform=ax.transAxes,fontsize=10,
        #     horizontalalignment='center',verticalalignment='center')
        ax.text(-0.1,0.5,str(labels[sim_reorder[i]-1]),transform=ax.transAxes,fontsize=20,
            horizontalalignment='center',verticalalignment='center')
        if sim_reorder[i]-1==7:
            ax.set_xlabel('Time (min)',x=0.5,horizontalalignment='center',fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=15)
        else:
            ax.set_xticklabels('')

        # plt.text(0.5,0.95,sys.argv[1],fontsize=15,transform=plt.gcf().transFigure,horizontalalignment='center')
    for i in range(8):
        row = sim_reorder[i]-1
        # ax = plt.subplot2grid((len(trap_num_list),1),(i,0))
        ax = plt.subplot2grid((8,1),(row,0))
        ax.set_ylim([0,max(peak_counts)])
        ax.set_yticks([ax.get_yticks()[0],ax.get_yticks()[-1]])




    file_name = 'trap_time_course_by_wind'
    # file_name = 'trap_time_course_by_wind_adjusted_fit_gauss'

    # png_file_name = file_name +'_wind_mag_'+str(wind_mag)+'_wind_angle_'+str(wind_angle)[0:4]
    png_file_name = file_name +'_wind_mag_'+str(wind_mag)#+'_wind_angle_'+str(wind_angle)[0:4]
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

# angles = list(np.linspace(3*np.pi/2,7*np.pi/4,6))
angles = [7*np.pi/4]#list(np.linspace(3*np.pi/2,7*np.pi/4,6))
iterations = range(num_iterations)

# angles = [np.pi,np.pi/2]

# f(angles[0],1)

# with poolcontext(processes=10) as pool:
#      pool.map(f_unpack, product(angles, iterations))

pool = multiprocessing.Pool(processes=10)
#
pool.map(g, angles)

# wind_mags = np.arange(0.4,3.8,0.2)
#wind_mags = [0.8,1.2,1.6,1.8,2.0]

#pool.map(g, wind_mags)

# g(wind_mags[0])

# g(angles[0])

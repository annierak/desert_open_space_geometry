import time
import scipy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animate
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
# from pompy import models,processors

from multiprocessing import Pool

def g(cast_delay):

    wind_angle = 9*scipy.pi/8.
    wind_mag = 1.6

    file_name = 'trap_arrival_by_wind_live_coarse_dt'
    file_name = file_name +'_wind_mag_'+str(wind_mag)
    # file_name = file_name +'_detection_threshold_'+str(detection_threshold)
    # file_name = file_name +'_cast_timeout_'+str(cast_timeout)
    # file_name = file_name +'_cast_interval_'+str(cast_interval)
    file_name = file_name +'_cast_delay_'+str(cast_delay)
    output_file = file_name+'.pkl'

    with open(output_file, 'r') as f:
        (_,swarm) = pickle.load(f)
        # swarm = pickle.load(f)

    #Trap arrival plot
    trap_locs = (2*scipy.pi/swarm.num_traps)*scipy.array(swarm.list_all_traps())
    sim_trap_counts = swarm.get_trap_counts()

    #Set 0s to 1 for plotting purposes
    sim_trap_counts[sim_trap_counts==0] = .5

    radius_scale = 0.3
    plot_size = 1.5
    # fig=plt.figure(200+int(10*wind_mag))
    fig=plt.figure()
    ax = plt.subplot(aspect=1)
    trap_locs_2d = [(scipy.cos(trap_loc),scipy.sin(trap_loc)) for trap_loc in trap_locs]
    patches = [plt.Circle(center, size) for center, size in zip(trap_locs_2d, radius_scale*sim_trap_counts/max(sim_trap_counts))]
    coll = matplotlib.collections.PatchCollection(patches, facecolors='blue',edgecolors='blue')
    ax.add_collection(coll)
    ax.set_ylim([-plot_size,plot_size]);ax.set_xlim([-plot_size,plot_size])
    ax.set_xticks([])
    ax.set_xticklabels('')
    ax.set_yticks([])
    ax.set_yticklabels('')

    biggest_trap_loc = trap_locs_2d[
    np.where(sim_trap_counts==max(sim_trap_counts))[0][0]]

    ax.text(biggest_trap_loc[0],biggest_trap_loc[1],
        str(int(max(sim_trap_counts))),horizontalalignment='center',
            verticalalignment='center',fontsize=18,color='white')

    #Wind arrow
    plt.arrow(0.5, 0.5, 0.1*scipy.cos(wind_angle), 0.1*scipy.sin(wind_angle),transform=ax.transAxes,color='b',
        width=0.01,head_width=0.05)
    # ax.text(0.55, 0.5,'Wind',transform=ax.transAxes,color='b')
    ax.text(0,1.5,'N',horizontalalignment='center',verticalalignment='center',fontsize=25)
    ax.text(0,-1.5,'S',horizontalalignment='center',verticalalignment='center',fontsize=25)
    ax.text(1.5,0,'E',horizontalalignment='center',verticalalignment='center',fontsize=25)
    ax.text(-1.5,0,'W',horizontalalignment='center',verticalalignment='center',fontsize=25)
    # plt.title('Simulated')
    fig.patch.set_facecolor('white')
    plt.axis('off')
    ax.text(0,1.7,'Trap Counts'+' (Wind Mag: '+str(wind_mag)[0:3]+')',horizontalalignment='center',verticalalignment='center',fontsize=20)
    plt.savefig(file_name+'.png',format='png')

pool = Pool(processes=6)
# pool.map(g,[0.025,0.075,0.1,0.125,0.15,0.175,0.2,0.225])
# pool.map(g,[1,10,15,40,60,100])
pool.map(g,[0.5,3,5,10,20,40]) #cast delay

# pool.map(g,
#     [[0.5,1.5],
#     [1,3],
#     [2,6],
#     [4,12],
#     [8,24],
#     [10,30],
#     [20,60]])

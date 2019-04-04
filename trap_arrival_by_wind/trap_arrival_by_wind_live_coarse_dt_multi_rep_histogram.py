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

num_iterations = 40


def g(wind_angle,num_iterations=num_iterations):

    # wind_angle = 7*np.pi/4
    wind_mag = 1.4

    #Loop through pickle files for that parameter value and collect counts

    for i in range(num_iterations):
        # file_name = 'trap_arrival_by_wind_live_coarse_dt'
        file_name = 'trap_time_course_by_wind'
        # file_name = 'trap_arrival_by_wind_fit_gauss'

        file_name = file_name +'_wind_mag_'+str(wind_mag)+'_wind_angle_'+str(wind_angle)[0:4]+'_iter_'+str(i)
        # file_name = file_name +'_wind_mag_'+str(wind_mag)

        output_file = file_name+'.pkl'

        with open(output_file, 'r') as f:
            (_,swarm) = pickle.load(f)
            # swarm = pickle.load(f)

        if i==0:
            sim_trap_counts = np.zeros((num_iterations,swarm.num_traps))

        sim_trap_counts[i,:] = swarm.get_trap_counts()

        #Set 0s to 0.5 for plotting purposes
        sim_trap_counts[i,:][sim_trap_counts[i,:]==0] = .5

    cross_trial_max = np.max(sim_trap_counts)
    max_trial, max_trap = np.vstack(np.where(sim_trap_counts==np.max(sim_trap_counts)))[:,0]
    mean_trap_counts = np.mean(sim_trap_counts,axis=0)
    trap_stds = np.std(sim_trap_counts,axis=0)
    for i in range(num_iterations):
        if i==0:
            trap_locs = (2*scipy.pi/swarm.num_traps)*scipy.array(swarm.list_all_traps())
            radius_scale = 0.3
            plot_size = 1.5
            fig=plt.figure(200+int(10*wind_mag))
            ax = plt.subplot(aspect=1)
            trap_locs_2d = [(scipy.cos(trap_loc),scipy.sin(trap_loc)) for trap_loc in trap_locs]
            ax.set_ylim([-plot_size,plot_size]);ax.set_xlim([-plot_size,plot_size])
            ax.set_xticks([])
            ax.set_xticklabels('')
            ax.set_yticks([])
            ax.set_yticklabels('')
            #Wind arrow
            plt.arrow(0.5, 0.5, 0.1*scipy.cos(wind_angle), 0.1*scipy.sin(wind_angle),transform=ax.transAxes,color='b',
                width=0.001, head_width=0.03, head_length=0.05)
            # ax.text(0.55, 0.5,'Wind',transform=ax.transAxes,color='b')
            ax.text(0,1.5,'N',horizontalalignment='center',verticalalignment='center',fontsize=25)
            ax.text(0,-1.5,'S',horizontalalignment='center',verticalalignment='center',fontsize=25)
            ax.text(1.5,0,'E',horizontalalignment='center',verticalalignment='center',fontsize=25)
            ax.text(-1.5,0,'W',horizontalalignment='center',verticalalignment='center',fontsize=25)
            fig.patch.set_facecolor('white')
            plt.axis('off')
            ax.text(0,1.7,'Trap Counts'+' (Wind Mag: '+str(wind_mag)[0:3]+')',horizontalalignment='center',verticalalignment='center',fontsize=20)


        patches = [plt.Circle(
            center, size) for center, size in zip(
                trap_locs_2d, radius_scale*sim_trap_counts[i,:]/cross_trial_max)]
        coll = matplotlib.collections.PatchCollection(
            patches, facecolors='none',edgecolors='blue',alpha=0.05)
        ax.add_collection(coll)
        if i==max_trial:
            biggest_trap_loc = trap_locs_2d[max_trap]
            #Label max trap count
            ax.text(biggest_trap_loc[0],biggest_trap_loc[1]+0.25,
                str(int(cross_trial_max)),horizontalalignment='center',
                    verticalalignment='center',fontsize=14,color='black')
        if i==num_iterations-1:
            patches = [plt.Circle(
                center, size) for center, size in zip(
                    trap_locs_2d, radius_scale*mean_trap_counts/cross_trial_max)]
            coll = matplotlib.collections.PatchCollection(
                patches, facecolors='none',edgecolors='red',alpha=0.5)
            ax.add_collection(coll)

            for j in range(swarm.num_traps):
                ax.text(trap_locs_2d[j][0],trap_locs_2d[j][1]-0.25,
                    str(trap_stds[j])[0:4

                    ],horizontalalignment='center',
                        verticalalignment='center',fontsize=8,color='blue')
                ax.text(trap_locs_2d[j][0]+0.25,trap_locs_2d[j][1],
                    str(mean_trap_counts[j])[0:4],horizontalalignment='center',
                        verticalalignment='center',fontsize=8,color='red')

            file_name = 'trap_arrival_histogram_by_wind_live_coarse_dt'
            png_file_name = file_name +'_wind_mag_'+str(wind_mag)+'_wind_angle_'+str(wind_angle)[0:4]
            plt.savefig(png_file_name+'.png',format='png',dpi=500)

    #Second plot:
    #Raster plot histogram of opposing trap pairs
    opp_pairs = utility.fold_across_axis(range(swarm.num_traps),wind_angle) #List of opposing trap pair tuples folded across wind
    #opp_pairs = utility.fold_across_axis(range(8),1*np.pi/6) #List of opposing trap pair tuples folded across wind

    plt.figure()
    gs = matplotlib.gridspec.GridSpec(swarm.num_traps/2,2)

    print(np.unique(sim_trap_counts,axis=0))

    for i in range(swarm.num_traps/2):
        ax = plt.subplot(gs[2*i%(swarm.num_traps/2),int(
            i*2>=swarm.num_traps/2)])
        trap1 = opp_pairs[i][0]
        n,bins,_= plt.hist(sim_trap_counts[:,trap1],bins=20)
        ax = plt.subplot(gs[(2*i+1)%(swarm.num_traps/2),int(
            (2*i+1)>=swarm.num_traps/2)])
        trap2 = opp_pairs[i][1]
        n,bins,_= plt.hist(sim_trap_counts[:,trap2],bins=20)
    plt.show()
    raw_input()


import multiprocessing
from itertools import product
from contextlib import contextmanager


# angles = list(np.linspace(3*np.pi/2,7*np.pi/4,6))
angles = [7*np.pi/8]#list(np.linspace(3*np.pi/2,7*np.pi/4,6))
iterations = range(num_iterations)
wind_mags = [0.8,1.2,1.6,1.8,2.0]
# angles = [np.pi,np.pi/2]

pool = multiprocessing.Pool(processes=6)

# pool.map(g, angles)
g(angles[0])
# pool.map(g, wind_mags)
# g(wind_mags[0])

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
from multiprocessing import Pool



def main(plume_width_factor):

    file_name = 'plume_width_testing'
    output_file = file_name+'.pkl'
    file_name = file_name +'plume_width_factor'+str(plume_width_factor)

    dt = 0.25
    plume_dt = 0.25
    frame_rate = 20
    times_real_time = 20 # seconds of simulation / sec in video
    capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))

    simulation_time = 2.*60. #seconds
    release_delay = 30.*60#/(wind_mag)

    t_start = 0.0
    t = 0. - release_delay

    # Set up figure
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111)

    wind_mag = 1.8
    wind_angle = 13*scipy.pi/8.

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
    centre_rel_diff_scale = plume_width_factor*2.
    # puff_release_rate = 0.001
    puff_release_rate = 10
    puff_spread_rate=0.005
    puff_init_rad = 0.01
    max_num_puffs=int(2e5)
    # max_num_puffs=100

    plume_model = models.PlumeModel(
        sim_region, source_pos, wind_field,simulation_time+release_delay,plume_dt,
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

    #Grab wind info to determine heading mean
    wind_x,wind_y = wind_mag*scipy.cos(wind_angle),wind_mag*scipy.sin(wind_angle)

    beta = 1.
    release_times = scipy.random.exponential(beta,(swarm_size,))
    kappa = 2.

    heading_data=None
    # xmin,xmax,ymin,ymax = -1000,1000,-1000,1000


    #Initial concentration plotting
    conc_array = array_gen.generate_single_array(plume_model.puffs)
    xmin = sim_region.x_min; xmax = sim_region.x_max
    ymin = sim_region.y_min; ymax = sim_region.y_max
    im_extents = (xmin,xmax,ymin,ymax)
    vmin,vmax = 0.,50.
    cmap = matplotlib.colors.ListedColormap(['white', 'orange'])
    conc_im = ax.imshow(conc_array.T[::-1], extent=im_extents,
    vmin=vmin, vmax=vmax, cmap=cmap)

    xmin,xmax,ymin,ymax = -1000,1000,-1000,1000

    buffr = 100
    ax.set_xlim((xmin-buffr,xmax+buffr))
    ax.set_ylim((ymin-buffr,ymax+buffr))


    #Conc array gen to be used for the flies
    sim_region_tuple = plume_model.sim_region.as_tuple()
    box_min,box_max = sim_region_tuple[1],sim_region_tuple[2]

    #Put the time in the corner
    (xmin,xmax) = ax.get_xlim();(ymin,ymax) = ax.get_ylim()
    text = '0 min 0 sec'
    timer= ax.text(xmax,ymax,text,color='r',horizontalalignment='right')
    ax.text(1.,1.02,'time since release:',color='r',transform=ax.transAxes,
        horizontalalignment='right')

    # #traps
    for x,y in traps.param['source_locations']:

        #Black x
        plt.scatter(x,y,marker='x',s=50,c='k')

        # Red circles
        # p = matplotlib.patches.Circle((x, y), 15,color='red')
        # ax.add_patch(p)

    #Remove plot edges and add scale bar
    fig.patch.set_facecolor('white')
    plt.plot([-900,-800],[900,900],color='k')#,transform=ax.transData,color='k')
    ax.text(-900,820,'100 m')
    plt.axis('off')


    # plt.ion()
    # plt.show()
    # raw_input()
    while t<simulation_time:
        for k in range(capture_interval):
            #update flies
            print('t: {0:1.2f}'.format(t))
            #update the swarm
            for j in range(int(dt/plume_dt)):
                wind_field.update(plume_dt)
                plume_model.update(plume_dt,verbose=True)
            t+= dt
        # Update live display
            if t>-10.*60.:
                conc_array = array_gen.generate_single_array(plume_model.puffs)

                # non_inf_log =
                log_im = scipy.log(conc_array.T[::-1])
                cutoff_l = scipy.percentile(log_im[~scipy.isinf(log_im)],10)
                cutoff_u = scipy.percentile(log_im[~scipy.isinf(log_im)],99)

                # im = (log_im>cutoff_l) & (log_im<0.1)
                # n = matplotlib.colors.Normalize(vmin=0,vmax=1)
                # image.set_data(im)
                # image.set_norm(n)

                conc_im.set_data(log_im)
                n = matplotlib.colors.Normalize(vmin=cutoff_l,vmax=cutoff_u)
                conc_im.set_norm(n)
                plt.savefig(file_name+'.png',format='png')

                t = simulation_time

    # writer.finish()
#
pool = Pool(processes=6)
pool.map(main,[0.25,1,4,16])

# main(1.)

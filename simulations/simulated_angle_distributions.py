#Given inputted behavioral parameters, run a simulation and save a hdf5 file
#with the fly density over space per unit time

import time
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib
matplotlib.use("Agg")
import sys
import itertools
import json
import cPickle as pickle
from matplotlib.animation import FuncAnimation
import datetime

import odor_tracking_sim.swarm_models as swarm_models
import odor_tracking_sim.wind_models as wind_models
import odor_tracking_sim.trap_models as trap_models
import odor_tracking_sim.odor_models as odor_models
import odor_tracking_sim.utility as utility
import odor_tracking_sim.simulation_running_tools as srt

file_name = 'simulated_angle_distributions'
output_file = file_name+'.pkl'

dt = 0.25
plot_period = 4 #In units of dt

release_delay = 0.

ending_radius = 20.
ending_fraction = 0.9 #fraction of flies past ending_radius at which to stop simulation

#wind
counter = 0
wind_mags = np.arange(0.4,3.8,0.2)
# wind_mags = [2.55]
for wind_mag in wind_mags:
    fig = plt.figure(counter+200)
    plt.close(fig)
    counter+=1
    wind_angle = scipy.pi/4.
    wind_param = {
                'speed': wind_mag,
                'angle': wind_angle,
                'evolving': False,
                'wind_dt': None,
                'dt': dt
                }
    wind_field = wind_models.WindField(param=wind_param)

    #traps
    number_sources = 8
    radius_sources = 1000.0
    trap_radius = 0.5
    location_list, strength_list = utility.create_circle_of_sources(number_sources,
                    radius_sources,0.)
    trap_param = {
            'source_locations' : location_list,
            'source_strengths' : strength_list,
            'epsilon'          : 0.01,
            'trap_radius'      : trap_radius,
            'source_radius'    : radius_sources
    }

    traps = trap_models.TrapModel(trap_param)

    plotting=True

    #Odor field--only because the swarm is currently programmed to require one
    odor_param = {
            'wind_field'       : wind_field,
            'diffusion_coeff'  :  0.25,
            'source_locations' : traps.param['source_locations'],
            'source_strengths' : traps.param['source_strengths'],
            'epsilon'          : 0.01,
            'trap_radius'      : traps.param['trap_radius']
            }
    odor_field = odor_models.FakeDiffusionOdorField(traps,param=odor_param)

    #Setup fly swarm
    wind_slippage = (0.,1.)
    swarm_size=10000
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
                'upper': 0.001
                },
            'schmitt_trigger':False,
            'low_pass_filter_length':3, #seconds
            'dt_plot': plot_period*dt,
            't_stop':3000.,
            'airspeed_saturation':True
            }

    swarm = swarm_models.BasicSwarmOfFlies(wind_field,traps,param=swarm_param,
        start_type='fh',track_plume_bouts=False,track_arena_exits=False)

    xmin,ymin,xmax,ymax = -100,-100,100,100
    plt.ion()
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111)
    ax.set_xlim([xmin,xmax]);ax.set_ylim([ymin,ymax])

    plt.plot(0,0,'o',color='red')
    fly_dots = plt.scatter(swarm.x_position, swarm.y_position,alpha=0.2)

    #Put the time in the corner
    text = '0 min 0 sec'
    timer= ax.text(xmax,ymax,text,color='r',horizontalalignment='right')

    t = 0.0

    while (np.sum(swarm.distance_to_origin>=ending_radius)/swarm_size)<ending_fraction:
        for k in range(plot_period):
            #update flies
            print('t: {0:1.2f}'.format(t))
            #update the swarm
            swarm.update(t,dt,wind_field,odor_field,traps)
            t+= dt
            time.sleep(.001)

        # Update plot
        fly_dots.set_offsets(scipy.c_[swarm.x_position,swarm.y_position])

        #Update time display
        text ='{0} min {1} sec'.format(int(scipy.floor(t/60.)),int(scipy.floor(t%60.)))
        timer.set_text(text)

        plt.pause(0.01)

    plt.close(fig)
    vfunc = scipy.vectorize(utility.cartesian_to_polar)
    xvels,yvels = swarm.x_position,swarm.y_position
    _,thetas = vfunc(xvels,yvels)
    output = (thetas)%(2*scipy.pi)

    plt.figure(int(wind_mag/0.4)+100)
    ax=plt.subplot(2,1,1,projection='polar')
    n,bins,_ = plt.hist(output%(2*np.pi),bins=500)
    ax.cla()
    plt.plot(bins,np.concatenate((n,[n[0]])))
    ax.set_ylim([0,np.max(n)])
    ax.set_yticks([])
    plt.title('Final Angle Histogram: Windspeed = '+str(wind_mag)[0:3],x=0.5,y=1.1)
    plt.xticks(np.linspace(np.pi/2,2*np.pi,4),('$\pi/2$','$\pi$','$3\pi/2$','$2\pi$'))

    ax = plt.subplot(2,1,2);plt.title('Final Angle CDF')
    cum,bins,_ = plt.hist(output%(2*np.pi),bins=500,cumulative=True)
    cum = cum/len(output)
    ax.cla()
    plt.plot(bins[:-1],cum)
    plt.xticks(np.linspace(np.pi/2,2*np.pi,4),('$\pi/2$','$\pi$','$3\pi/2$','$2\pi$'))
    # ax.set_ylim([0,np.max(n)])
    ax.set_yticks([])
    plt.xlim([0,2*np.pi])

plt.show()
raw_input()

with open(output_file, 'w') as f:
    pickle.dump(swarm,f)

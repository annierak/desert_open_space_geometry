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

for wind_mag in [0.8]: #np.arange(0.4,3.8,0.2):

    file_name = 'trap_arrival_by_wind_sutton'
    file_name = file_name +'_wind_mag_'+str(wind_mag)
    output_file = file_name+'.pkl'

    dt = 0.25
    frame_rate = 20
    times_real_time = 20 # seconds of simulation / sec in video
    capture_interval = int(scipy.ceil(times_real_time*(1./frame_rate)/dt))

    simulation_time = 50.*60. #seconds
    release_delay = 0.*60#/(wind_mag)

    t_start = 0.0
    t = 0. - release_delay



    # Set up figure
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111)

    #Video
    FFMpegWriter = animate.writers['ffmpeg']
    metadata = {'title':file_name,}
    writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
    writer.setup(fig, file_name+'.mp4', 500)

    wind_angle = 7*scipy.pi/4.
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
    im_extents = xlim[0], xlim[1], ylim[0], ylim[1]

    source_pos = scipy.array([scipy.array(tup) for tup in traps.param['source_locations']])

    # Set up Sutton plume model -- use the same parameters as determined in
    # fly_behavior_sim/near_plume_simulation_sutton.py

    Q,C_y,n = (5.,0.4,0.8)

    suttonPlumes = models.SuttonModelPlume(Q,C_y,n,source_pos,wind_angle)

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

    #Flies also use parameters (for schmitt_trigger, detection probabilities)
    # determined in
    #fly_behavior_sim/near_plume_simulation_sutton.py

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
                'upper': 1
                },
            'odor_probabilities'  : {
            'lower': 0.9,    # detection probability/sec of exposure
            'upper': 0.8,  # detection probability/sec of exposure
            },
            'schmitt_trigger':True,
            'low_pass_filter_length':3, #seconds
            'dt_plot': capture_interval*dt,
            't_stop':3000.,
            'cast_timeout':20,
            'airspeed_saturation':True
            }

    swarm = swarm_models.BasicSwarmOfFlies(wind_field,traps,param=swarm_param,
        start_type='fh',track_plume_bouts=False,track_arena_exits=False)

    # xmin,xmax,ymin,ymax = -1000,1000,-1000,1000


    #Concentration plotting
    conc_d = suttonPlumes.conc_im(im_extents)

    cmap = matplotlib.colors.ListedColormap(['white', 'orange'])

    conc_im = plt.imshow(conc_d,extent=im_extents,
        interpolation='none',cmap = cmap,origin='lower')

    log_im = scipy.log(conc_d)
    cutoff_l = scipy.percentile(log_im[~scipy.isinf(log_im)],10)
    cutoff_u = scipy.percentile(log_im[~scipy.isinf(log_im)],99)

    conc_im.set_data(log_im)
    n = matplotlib.colors.Normalize(vmin=cutoff_l,vmax=cutoff_u)
    conc_im.set_norm(n)

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
    fly_facecolors =  [facecolor_dict[mode] for mode in swarm.mode]
    fly_dots = plt.scatter(swarm.x_position, swarm.y_position,
        edgecolor=fly_edgecolors,facecolor = fly_facecolors,alpha=0.9)

    #Put the time in the corner
    (xmin,xmax) = ax.get_xlim();(ymin,ymax) = ax.get_ylim()
    text = '0 min 0 sec'
    timer= ax.text(xmax,ymax,text,color='r',horizontalalignment='right')
    ax.text(1.,1.02,'time since release:',color='r',transform=ax.transAxes,
        horizontalalignment='right')

    #Wind arrow
    plt.arrow(0.7, 0.9, 0.07, -0.07,transform=ax.transAxes,color='b',
        width=0.001)
    ax.text(0.75, 0.9,'Wind',transform=ax.transAxes,color='b')


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

    #Fly behavior color legend
    for mode,fly_facecolor,fly_edgecolor,a in zip(
        ['Dispersing','Surging','Casting','Trapped'],
        facecolor_dict.values(),
        edgecolor_dict.values(),
        [0,50,100,150]):

        plt.scatter([1000],[-600-a],edgecolor=fly_edgecolor,
        facecolor=fly_facecolor, s=20)
        plt.text(1050,-600-a,mode,verticalalignment='center')


    plt.ion()
    # plt.show()
    # raw_input()
    while t<simulation_time:
        for k in range(capture_interval):
            #update flies
            print('t: {0:1.2f}'.format(t))
            swarm.update(t,dt,wind_field,suttonPlumes,traps)
            #Update time display
            release_delay = release_delay/60.
            text ='{0} min {1} sec'.format(
                int(scipy.floor(abs(t/60.))),int(scipy.floor(abs(t)%60.)))
            timer.set_text(text)
            t+= dt
            time.sleep(0.001)
        # Update live display
        '''plot the flies'''
        fly_dots.set_offsets(scipy.c_[swarm.x_position,swarm.y_position])

        fly_edgecolors = [edgecolor_dict[mode] for mode in swarm.mode]
        fly_facecolors =  [facecolor_dict[mode] for mode in swarm.mode]
        #
        fly_dots.set_edgecolor(fly_edgecolors)
        fly_dots.set_facecolor(fly_facecolors)

        trap_list = []
        for trap_num, trap_loc in enumerate(traps.param['source_locations']):
            mask_trap = swarm.trap_num == trap_num
            trap_cnt = mask_trap.sum()
            trap_list.append(trap_cnt)
        total_cnt = sum(trap_list)

        plt.pause(0.0001)
        writer.grab_frame()

    writer.finish()

    with open(output_file, 'w') as f:
        pickle.dump(swarm,f)

    #Trap arrival plot
    trap_locs = (2*scipy.pi/swarm.num_traps)*scipy.array(swarm.list_all_traps())
    sim_trap_counts = swarm.get_trap_counts()

    #Set 0s to 1 for plotting purposes
    sim_trap_counts[sim_trap_counts==0] = .5

    radius_scale = 0.3
    plot_size = 1.5
    plt.figure(200+int(10*wind_mag))
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
    ax.text(0,1.5,'N',horizontalalignment='center',verticalalignment='center',fontsize=25)
    ax.text(0,-1.5,'S',horizontalalignment='center',verticalalignment='center',fontsize=25)
    ax.text(1.5,0,'E',horizontalalignment='center',verticalalignment='center',fontsize=25)
    ax.text(-1.5,0,'W',horizontalalignment='center',verticalalignment='center',fontsize=25)
    # plt.title('Simulated')
    fig.patch.set_facecolor('white')
    plt.axis('off')
    ax.text(0,1.7,'Trap Counts'+' (Wind Mag: '+str(wind_mag)[0:3],horizontalalignment='center',verticalalignment='center',fontsize=20)
    plt.savefig(file_name+'.png',format='png')

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

def main(wind_mag,i):#np.arange(0.4,3.8,0.2):

    random_state = np.random.RandomState(i)

    file_name = 'test_lazy_plumes_single_plume_wind_mag_'+str(wind_mag)

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

    # #Video
    FFMpegWriter = animate.writers['ffmpeg']
    metadata = {'title':file_name,}
    writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
    writer.setup(fig, file_name+'.mp4', 500)


    wind_angle = 0.
    wind_param = {
                'speed': wind_mag,
                'angle': wind_angle,
                'evolving': False,
                'wind_dt': None,
                'dt': dt
                }
    wind_field_noiseless = wind_models.WindField(param=wind_param)

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

    #Odor arena
    xlim = (0., 1800.)
    ylim = (-500., 500.)
    sim_region = models.Rectangle(xlim[0], ylim[0], xlim[1], ylim[1])

    wind_region = models.Rectangle(xlim[0]*2,ylim[0]*2,
    xlim[1]*2,ylim[1]*2)
    im_extents = xlim[0], xlim[1], ylim[0], ylim[1]

    #lazy plume parameters
    puff_mol_amount = 1.
    r_sq_max=20;epsilon=0.00001;N=1e6


    lazyPompyPlumes = models.OnlinePlume(sim_region, source_pos, wind_field_noiseless,
        simulation_time,dt,r_sq_max,epsilon,puff_mol_amount,N)



    # Concentration plotting
    # conc_d = lazyPompyPlumes.conc_im(im_extents)
    #
    # cmap = matplotlib.colors.ListedColormap(['white', 'orange'])
    # cmap = 'YlOrBr'
    #
    # conc_im = plt.imshow(conc_d,extent=im_extents,
    #     interpolation='none',cmap = cmap,origin='lower')
    #
    # plt.colorbar()
    #
    #
    #
    # # #traps
    # for x,y in traps.param['source_locations']:
    #
    #     #Black x
    #     plt.scatter(x,y,marker='x',s=50,c='k')
    #
    # plt.ion()
    # plt.show()

    while t<simulation_time:
        for k in range(capture_interval):
            #update flies
            print('t: {0:1.2f}'.format(t))
            x_locs,y_locs = np.linspace(0., 1800.,1000),np.random.uniform(-500., 500.,1000)
            lazyPompyPlumes.value(x_locs,y_locs)
            raw_input()
            t+= dt
            # time.sleep(0.001)
        # Update live display
        # Update time display
        release_delay = release_delay/60.
        text ='{0} min {1} sec'.format(
            int(scipy.floor(abs(t/60.))),int(scipy.floor(abs(t)%60.)))
        timer.set_text(text)
        #
        '''plot the flies'''
        fly_dots.set_offsets(scipy.c_[swarm.x_position,swarm.y_position])

        fly_edgecolors = [edgecolor_dict[mode] for mode in swarm.mode]
        fly_facecolors =  [facecolor_dict[mode] for mode in swarm.mode]
        #
        fly_dots.set_edgecolor(fly_edgecolors)
        fly_dots.set_facecolor(fly_facecolors)
        plt.pause(0.0001)
        writer.grab_frame()

        trap_list = []
        for trap_num, trap_loc in enumerate(traps.param['source_locations']):
            mask_trap = swarm.trap_num == trap_num
            trap_cnt = mask_trap.sum()
            trap_list.append(trap_cnt)
        total_cnt = sum(trap_list)


    # writer.finish()

    with open(output_file, 'w') as f:
        pickle.dump((wind_field,swarm),f)


from multiprocessing import Pool

# pool = Pool(processes=6)
mags = [0.8,1.2,1.6,1.8,2.0]
# mags = [1.4]

main(1.2,1)

# pool.map(main,mags)

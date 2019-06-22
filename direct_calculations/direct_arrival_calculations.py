import time
import scipy
import matplotlib.pyplot as plt
import matplotlib
import sys
import itertools
import h5py
import json
import cPickle as pickle
import numpy as np
import odor_tracking_sim.utility as utility
from pompy import models


#First, do a single relase time. Next iteration will incorporate release delays.
#First draft, use the LogisticPlumeModel object for the plumes.

wind_angle = 7*scipy.pi/8.
wind_mag = 1.6
# wind_mag = 2.4
# wind_mag = 0.6
# num_flies = 20000
# num_flies = 5
num_flies = 20000
fly_speed = 1.6

release_times=0.

K = -.4
x_0 = 300

# K = -1.
# x_0 = 1000

number_sources = 8
radius_sources = 1000.0
source_locations, _ = utility.create_circle_of_sources(number_sources,
                radius_sources,None)
source_pos = scipy.array([scipy.array(tup) for tup in source_locations])
release_location = np.zeros(2)

intended_heading_angles = np.random.uniform(0,2*np.pi,num_flies)
intended_heading_angles = np.linspace(0,2*np.pi,num_flies)
# intended_heading_angles = np.radians(np.array([190,350]))

# Set up logistic prob plume object

logisticPlumes = models.LogisticProbPlume(K,x_0,source_pos,wind_angle)


#For visualization purposes for testing
xlim = (-1500., 1500.)
ylim = (-1500., 1500.)
im_extents = xlim[0], xlim[1], ylim[0], ylim[1]

gaussianfitPlumes = models.GaussianFitPlume(source_pos,wind_angle,wind_mag)
conc_im = gaussianfitPlumes.conc_im(im_extents,samples=200)
# conc_im = logisticPlumes.conc_im(im_extents,samples=200)
plt.figure(1)
plt.imshow(conc_im,extent=im_extents,aspect='equal',origin='lower',cmap='bone_r')

for x,y in source_locations:

    #Black x
    plt.scatter(x,y,marker='x',s=50,c='k')



def f0(intended_heading_angles):
    #Converts intended heading angles to track heading angles
    #Currently only have computation for c1 = 0, c2 = 1
    n = num_flies
    intended_heading_angles = np.linspace(360./n,360,n)*np.pi/180
    signed_wind_par_mags = wind_mag*np.cos(wind_angle-intended_heading_angles)
    # print(signed_wind_par_mags)
    # inds_limit_affected = (wind_par_mags<-0.8) | (wind_par_mags>4.)
    # thetas_adjusted = np.copy(intended_heading_angles)
    # r_1s = fly_speed*np.ones(num_flies)
    # r_1s[wind_par_mags<-0.8] = wind_par_mags[wind_par_mags<-0.8]+2.4
    # r_1s[wind_par_mags>4.] = wind_par_mags[wind_par_mags>4.]-2.4
    # sign_change_inds = (r_1s<0.)
    # thetas_adjusted[sign_change_inds] = (
    #     intended_heading_angles[sign_change_inds]+np.pi)%(2*np.pi)
    # track_heading_angles = (np.arctan(wind_mag*np.sin(
    #     wind_angle-thetas_adjusted)/np.abs(r_1s))+thetas_adjusted)%(2*np.pi)
    adjusted_mag = utility.speed_sigmoid_func(signed_wind_par_mags)
    # print('-------------')
    # print(adjusted_mag)

    intended_heading_unit_vectors = np.vstack((
        np.cos(intended_heading_angles),np.sin(intended_heading_angles)))

    # print('-------------')
    # print(intended_heading_unit_vectors)


    intended_heading_vectors = adjusted_mag*intended_heading_unit_vectors

    w_vec = np.array([wind_mag*np.cos(wind_angle),wind_mag*np.sin(wind_angle)])
    # print('-------------')
    # print(w_vec)
    wind_par_vec = (np.dot(
        w_vec,intended_heading_unit_vectors))*intended_heading_unit_vectors
    # print('-------------')
    # print(wind_par_vec)

    w_perp_vec = w_vec[:,None] - wind_par_vec
    heading_final_vec = intended_heading_vectors+w_perp_vec
    dispersing_speeds = np.sqrt(np.sum(heading_final_vec**2,axis=0))
    track_heading_angles = np.arctan2(heading_final_vec[1],heading_final_vec[0])

    plt.figure()
    ax = plt.subplot()
    plt.scatter(heading_final_vec[0],heading_final_vec[1])
    plt.scatter(dispersing_speeds*np.cos(track_heading_angles),
        dispersing_speeds*np.sin(track_heading_angles))
    ax.set_aspect('equal')


    # plt.figure()
    # ax=plt.subplot(2,1,1,projection='polar')
    # n,bins,_ = plt.hist(track_heading_angles%(2*np.pi),bins=500)
    # ax.cla()
    # plt.plot(bins,np.concatenate((n,[n[0]])))
    # ax.set_yticks([])
    # plt.xticks(np.linspace(np.pi/2,2*np.pi,4),('$\pi/2$','$\pi$','$3\pi/2$','$2\pi$'))
    # plt.subplot(2,1,2)
    # plt.hist(dispersing_speeds)





    return track_heading_angles,dispersing_speeds

def f1(track_heading_angles,source_locations):
    #Convert track_heading_angles to a list of plume distances for each fly
    #(for example, the row for one fly would be [np.nan,500,700,800,np.nan,np.nan,np.nan,np.nan]
    # if the fly's future path intersected only plumes 2,3,4, at those distances) )

    #Using geometric algorithm here: http://geomalgorithms.com/a05-_intersect-1.html

    #Turn the track heading angles into unit vectors (v)
    fly_unit_vectors = np.vstack((np.cos(track_heading_angles),np.sin(track_heading_angles)))
    #shape is (2 x flies)

    #Turn the wind angle into a plume unit vector (u)
    plume_unit_vector = np.array([np.cos(wind_angle),np.sin(wind_angle)])
    #shape is (2)

    #Compute vector from release location to each source
    w = source_pos - release_location
    #shape is (traps x 2)

    #Compute s_1 & t_1

    #s_1 = (v2*w1-v1*w2)/(v1*u2-v2*u1)
    s_1 = (fly_unit_vectors[None,1,:]*w[:,0,None]-fly_unit_vectors[None,0,:]*w[:,1,None])/(
        fly_unit_vectors[None,0,:]*plume_unit_vector[None,1,None]-
            fly_unit_vectors[None,1,:]*plume_unit_vector[None,0,None])

    #t_1 = (u1*w2-u2*w1)/(u1*v2-u2*v1)
    t_1 = (plume_unit_vector[None,0,None]*w[:,1,None]-plume_unit_vector[None,1,None]*w[:,0,None])/(
        fly_unit_vectors[None,1,:]*plume_unit_vector[None,0,None]-
            fly_unit_vectors[None,0,:]*plume_unit_vector[None,1,None])

    #collapse extra axis
    s_1 = np.squeeze(s_1)

    #set all locations where s_1 or t_1 is negative to np.nan

    s_1[s_1<0.] = np.nan
    s_1[t_1<0.] = np.nan

    return s_1.T,t_1.T #shape is (n flies x n plumes), entry is the distance downwind of that plume

def f2(intersection_distances):
    #Convert intersection_distances to probabilities
    #using a specified distance-probability function


    success_probabilities = logisticPlumes.logistic_1d(intersection_distances)

    # plt.figure()
    #
    # plt.hist(intersection_distances[~np.isnan(intersection_distances)])
    #
    # plt.figure()
    # plt.plot(intersection_distances[
    #     (intersection_distances<1000.)].flatten(),success_probabilities[(
    #         intersection_distances<1000.)].flatten(),'o')
    # plt.show()

    return success_probabilities

def f3(success_probabilities,dispersal_distances):
    #Use the success_probabilities to determine (with a random draw)
    #which plume each fly will chase, if any
    #the input is shape (n flies x n plumes)

    #First, do a random draw for each plume it will intersect

    #Choose among the positive results of the draw (among possible plumes)
    #by choosing the plume that drew a 1 which has the shortest dispersal
    #distance for the fly

    # print('success_probabilities')
    # print(np.array2string(success_probabilities,precision=2))

    draws = np.random.binomial(1,success_probabilities,
        size=np.shape(success_probabilities)).astype(bool)

    # print('draws')
    # print(np.array2string(draws,precision=2))
    #
    # print('directions')
    # print(np.array2string(np.degrees(track_heading_angles),precision=0))
    #
    # print('dispersal_distances')
    # print(np.array2string(dispersal_distances,precision=2))
    #


    distances_with_positive_draws = np.full_like(success_probabilities,np.nan)

    distances_with_positive_draws[draws] = dispersal_distances[draws]

    #Check that the above value either has positive floats or nan's
    assert(np.sum(distances_with_positive_draws[~np.isnan(
        distances_with_positive_draws)]<0.)<1)

    #Take the inverse of each element to reverse the order and avoid 0 being the min
    distances_with_positive_draws[~np.isnan(
        distances_with_positive_draws)] = 1./(
            distances_with_positive_draws[~np.isnan(distances_with_positive_draws)])

    #Set the nan values to 0.
    distances_with_positive_draws[np.isnan(distances_with_positive_draws)] = 0.


    plume_assignments = distances_with_positive_draws.argmax(axis=1).astype(float)

    plume_assignments[np.sum(draws,axis=1)==0] = np.nan

    # print('plume_assignments')
    # print(np.array2string(plume_assignments,precision=3))

    #output is of shape (flies) which has for each fly number 0-7 of the traps
    #it goes to, or np.nan if it does not go to a trap
    return plume_assignments

def f4(plume_assignments,dispersal_distances,dispersing_speeds):
    #Now that we know which plume each fly is chasing,
    #use the intersection distances and plume ids to compute the time
    #it took for each fly to travel from the release site to
    #the plume it ended up detecting

    release_to_chosen_plume_distances = np.full(num_flies,np.nan)

    mask = ~np.isnan(plume_assignments)
    cols = plume_assignments[mask].astype(int)
    rows = np.where(mask)

    release_to_chosen_plume_distances[mask] = dispersal_distances[rows,cols].flatten()

    dispersal_travel_times = (release_to_chosen_plume_distances/dispersing_speeds)

    return dispersal_travel_times

def f5(plume_assignments,dispersal_travel_times,intersection_distances):
    #Use the intersection distances and plume ids to compute the time
    #that each fly arrived at the source of the plume it successfully chases
    intersection_distances_chosen_plume = np.full(num_flies,np.nan)

    mask = ~np.isnan(plume_assignments)
    cols = plume_assignments[mask].astype(int)
    rows = np.where(mask)[0]

    intersection_distances_chosen_plume[mask] = intersection_distances[rows,cols].flatten()

    chasing_times = (intersection_distances_chosen_plume/fly_speed)

    arrival_times = release_times+dispersal_travel_times+chasing_times

    return arrival_times[~np.isnan(arrival_times)],chasing_times,rows,cols

last = time.time()
#Convert intended heading angles to track heading angles
track_heading_angles,dispersing_speeds = f0(intended_heading_angles)
# raw_input()


if num_flies<50:
    plt.figure(1)
    for i in range(num_flies):
        plt.plot([0,1500*np.cos(track_heading_angles[i])],
            [0,1500*np.sin(track_heading_angles[i])],label='fly '+
                str(i)+', track '+str(np.degrees(track_heading_angles[i]))[0:4])
    plt.legend()

else:
    time = 5*60.
    mag = time*dispersing_speeds
    plt.figure(1)
    plt.scatter(mag*np.cos(track_heading_angles),mag*np.sin(track_heading_angles))
    # plt.show()

#Convert track_heading_angles to a list of plume intersection locations for each fly
intersection_distances,dispersal_distances = f1(track_heading_angles,source_locations)
#Convert intersection_distances to probabilities
success_probabilities = f2(intersection_distances)
# print('(flies x traps)')
# print(np.array2string(success_probabilities,precision=2))
#Convert probabilities to plume assignments
plume_assignments = f3(success_probabilities,dispersal_distances)

#Compute the time each fly intersected the plume it ended up detecting
dispersal_travel_times = f4(plume_assignments,dispersal_distances,dispersing_speeds)
#compute the time each fly arrived at the source of the plume it successfully chases
arrival_times,chasing_times,which_flies,which_traps = f5(plume_assignments,dispersal_travel_times,intersection_distances)

# sys.exit()
# print(which_traps)
# print(which_flies)

# for trap in range(8):
#     print(dispersal_travel_times[which_flies[which_traps==trap]])
#     print(chasing_times[which_flies[which_traps==trap]])
#     print(arrival_times[which_traps==trap])
#     print('--------')

# plt.show()

plt.figure()

num_bins = 50


peak_counts = scipy.zeros(8)
rasters = []

fig = plt.figure(figsize=(7, 11))

fig.patch.set_facecolor('white')

labels = ['N','NE','E','SE','S','SW','W','NW']

sim_reorder = scipy.array([3,2,1,8,7,6,5,4])

# plt.ion()

#Simulated histogram
# for i in range(len(trap_num_list)):
axes = []
for i in range(8):

    row = sim_reorder[i]-1
    ax = plt.subplot2grid((8,1),(row,0))
    t_sim = arrival_times[which_traps==i]

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
        ax.set_title('Cumulative Trap Arrivals \n  K: '+str(K)+', x_0: '+str(x_0))

    ax.set_xlim([0,50])
    # plt.pause(0.001)
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
    axes.append(ax)

for ax in axes:
    # row = sim_reorder[i]-1
    # ax = plt.subplot2grid((8,1),(row,0))
    # ax.set_ylim([0,max(peak_counts)])
    ax.set_ylim([0,400])
    ax.set_yticks([ax.get_yticks()[0],ax.get_yticks()[-1]])
    # raw_input()


plt.show()

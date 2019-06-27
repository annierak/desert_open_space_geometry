import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import odor_tracking_sim.utility as utility
from pompy import models
from matplotlib.widgets import Slider,Button
import sys

def f0(intended_heading_angles,wind_mag,wind_angle):
    #Converts intended heading angles to track heading angles
    #Currently only have computation for c1 = 0, c2 = 1
    n = len(intended_heading_angles)
    # intended_heading_angles = np.linspace(360./n,360,n)*np.pi/180
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

    # plt.figure()
    # ax = plt.subplot()
    # plt.scatter(heading_final_vec[0],heading_final_vec[1])
    # plt.scatter(dispersing_speeds*np.cos(track_heading_angles),
    #     dispersing_speeds*np.sin(track_heading_angles))
    # ax.set_aspect('equal')


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

def f1(track_heading_angles,source_locations,wind_angle,release_location=np.array([0,0])):
    #cone_angle: angle of spread from directly downwind
    #ie cone_angle = 10 degrees gives a 20 degree wedge

    #Convert track_heading_angles to a list of plume distances for each fly
    #(for example, the row for one fly would be [np.nan,500,700,800,np.nan,np.nan,np.nan,np.nan]
    # if the fly's future path intersected only plumes 2,3,4, at those distances) )

    #Using geometric algorithm here: http://geomalgorithms.com/a05-_intersect-1.html
    #referring to diagram under 'Non-Parallel Lines'

    #u - vector from source (P_0) downwind
    #v - heading vector originating from release location (Q_0)
    #w - vector from release location (Q_0) to source (P_0)
    #s_1 - distance along u from source (P_0) to intersection point
    #t_1 - distance along v from release_location (Q_0) to intersection point


    #Turn the track heading angles into unit vectors (v)
    fly_unit_vectors = np.vstack((np.cos(track_heading_angles),np.sin(track_heading_angles)))
    #shape is (2 x flies)

    #Turn the wind angle into a plume unit vector (u)
    plume_unit_vector = np.array([np.cos(wind_angle),np.sin(wind_angle)])
    #shape is (2)



    #Compute vector from release location to each source
    w = source_locations - release_location
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

    return s_1.T,t_1.T
    #s_1.T - shape is (n flies x n plumes), entry is the distance downwind of that plume
    #t_1.T - shape is (n flies x n plumes),
        #entry is the distance traveled from the release point to the intersection point

def f1_wedge(track_heading_angles,source_locations,wind_angle,
    cone_angle,release_location=np.array([0,0])):
    #Finds the intersection distances with edges of a wedge,
    #rather than a plume line

    #cone_angle: angle of spread from directly downwind
    #ie cone_angle = 10 degrees gives a 20 degree wedge

    #Convert track_heading_angles to a list of plume distances for each fly
    #(for example, the row for one fly would be [np.nan,500,700,800,np.nan,np.nan,np.nan,np.nan]
    # if the fly's future path intersected only plumes 2,3,4, at those distances) )

    #Using geometric algorithm here: http://geomalgorithms.com/a05-_intersect-1.html
    #referring to diagram under 'Non-Parallel Lines'

    #u - vector from source (P_0) downwind
    #v - heading vector originating from release location (Q_0)
    #w - vector from release location (Q_0) to source (P_0)
    #s_1 - distance along u from source (P_0) to intersection point
    #t_1 - distance along v from release_location (Q_0) to intersection point

    #Turn the track heading angles into unit vectors (v)
    fly_unit_vectors = np.vstack((np.cos(track_heading_angles),np.sin(track_heading_angles)))

    #shape is (2 x flies)
    #Turn (wind_angle-cone_angle,wind_angle+cone_angle) into a plume unit vector (u)
    plume_unit_vector = np.array(
        [[np.cos(wind_angle-cone_angle),np.sin(wind_angle-cone_angle)],
        [np.cos(wind_angle+cone_angle),np.sin(wind_angle+cone_angle)]])

    #shape is (cone sides (2) x 2)

    #Compute vector from release location to each source
    w = source_locations - release_location
    #shape is (traps x 2)

    #Compute s_1 & t_1

    #In the none-wedge version, the below have shape (traps x 2 x flies), and
    #then collapse to (traps x flies) after the computation.

    #For wedges, add an extra dimension to the front
    #for 'which cone side' (-cone_angle,cone_angle)

    #new shape is (2 x traps x 2 x flies)--> collapse to (2 x traps x flies)

    #s_1 = (v2*w1-v1*w2)/(v1*u2-v2*u1)
    s_1 = (fly_unit_vectors[None,None,1,:]*w[None,:,0,None]
        -fly_unit_vectors[None,None,0,:]*w[None,:,1,None])/(
            fly_unit_vectors[None,None,0,:]*plume_unit_vector[:,None,1,None]-
                fly_unit_vectors[None,None,1,:]*plume_unit_vector[:,None,0,None])

    #t_1 = (u1*w2-u2*w1)/(u1*v2-u2*v1)
    t_1 = (plume_unit_vector[:,None,0,None]*w[None,:,1,None]
        -plume_unit_vector[:,None,1,None]*w[None,:,0,None])/(
            fly_unit_vectors[None,None,1,:]*plume_unit_vector[:,None,0,None]-
                fly_unit_vectors[None,None,0,:]*plume_unit_vector[:,None,1,None])

    #collapse extra axis
    s_1 = np.squeeze(s_1) #now shape is (2 x traps x flies)

    #set all intersection values where s_1 or t_1 is negative to np.inf

    # fig10 = plt.figure(10)
    #
    # to_plot = np.concatenate((
    #     track_heading_angles[(s_1[1,3,:]>0.)],#&(s_1[1,3,:]>0.)],
    #     track_heading_angles[(s_1[1,3,:]>0.)]))  #&(s_1[1,3,:]>0.)]))
    #
    # n,bins = np.histogram(to_plot%(2*np.pi),bins=np.linspace(0,2*np.pi,50))
    #
    # try:
    #     global hist
    #     hist.set_ydata(n)
    #     hist.set_xdata(bins[:-1])
    # except(NameError):
    #     hist, = plt.plot(bins[:-1],n,'o')
    #
    # fig10.canvas.draw_idle()

    inds = (s_1>0.)&(t_1>0.)
    s_1[~inds] = np.inf
    t_1[~inds] = np.inf

    #For some reason the below does NOT have the identical effect as the above
    #for wide cone angles (>pi/8)

    # s_1[s_1<0.] = np.inf
    # t_1[s_1<0.] = np.inf
    # # s_1[t_1<0.] = np.inf
    # t_1[t_1<0.] = np.inf

    #collapse s_1 and t_1 along the 'which cone side' dimension
    #by selecting the cone side with the shorter (finite, positive)
    #t_1 (distance from release_location to intersection point)

    #first collapse t_1
    collapsed_t_1 = np.min(t_1,axis=0)

    where_no_intersections = np.isinf(collapsed_t_1) #value true for fly-trap pairs with no intersection

    #make cone_side_mask, shape (traps x flies) which is a mask (entries (0,1))
    #of which cone sides were intersected (nan if neither)
    cone_side_mask_rough = (t_1==np.min(t_1,axis=0))
    cone_side_mask = np.full(np.shape(collapsed_t_1),np.nan)
    cone_side_mask[cone_side_mask_rough[0]] = 0.
    cone_side_mask[cone_side_mask_rough[1]] = 1.
    cone_side_mask[where_no_intersections] = np.nan

    #now collapse s_1, the distance from source (P_0) to intersection point,
    #by selecting the assigned side (or none of them)

    collapsed_s_1 = np.full(np.shape(collapsed_t_1),np.nan)

    collapsed_s_1[(cone_side_mask==0.)] = s_1[
        0,np.where(cone_side_mask==0)[0],np.where(cone_side_mask==0)[1]]
    collapsed_s_1[(cone_side_mask==1.)] = s_1[
        1,np.where(cone_side_mask==1.)[0],np.where(cone_side_mask==1.)[1]]

    #Check that the shape of s_1 and t_1 are as expected

    return collapsed_s_1.T,collapsed_t_1.T
    #s_1.T - shape is (n flies x n plumes), entry is the distance downwind of that plume
    #t_1.T - shape is (n flies x n plumes),
        #entry is the distance traveled from the release point to the intersection point


def f2(intersection_distances,K,x_0,source_locations,wind_angle):
    #Convert intersection_distances to probabilities
    #using a specified distance-probability function

    logisticPlumes = models.LogisticProbPlume(K,x_0,source_locations,wind_angle)


    success_probabilities = logisticPlumes.logistic_1d(intersection_distances)
    # print(np.sum(np.isnan(intersection_distances[:,3])))



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
    #by **choosing the plume that drew a 1 which has the shortest dispersal
    #distance for the fly**

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

    # print('-----------------')
    # print(dispersal_distances[draws[:,0],0])
    # print(success_probabilities[draws[:,0],0])
    # print(np.sum(success_probabilities[draws[:,0],0]>0))
    # print(dispersal_distances[draws[:,3],3])
    # print(success_probabilities[:,3])
    # print(np.sum(success_probabilities[:,3]>0))


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
    # print('--------')
    # print(np.sum(plume_assignments==0))
    # print(np.sum(plume_assignments==3))


    return plume_assignments

def f4(plume_assignments,dispersal_distances,dispersing_speeds):
    #Now that we know which plume each fly is chasing,
    #use the intersection distances and plume ids to compute the time
    #it took for each fly to travel from the release site to
    #the plume it ended up detecting

    release_to_chosen_plume_distances = np.full(len(plume_assignments),np.nan)

    mask = ~np.isnan(plume_assignments)
    cols = plume_assignments[mask].astype(int)
    rows = np.where(mask)

    release_to_chosen_plume_distances[mask] = dispersal_distances[rows,cols].flatten()

    dispersal_travel_times = (release_to_chosen_plume_distances/dispersing_speeds)

    return dispersal_travel_times,release_to_chosen_plume_distances

def f5(plume_assignments,dispersal_travel_times,\
    intersection_distances,fly_speed,release_times):
    #Use the intersection distances and plume ids to compute the time
    #that each fly arrived at the source of the plume it successfully chases
    intersection_distances_chosen_plume = np.full(len(plume_assignments),np.nan)

    mask = ~np.isnan(plume_assignments)
    cols = plume_assignments[mask].astype(int)
    rows = np.where(mask)[0]

    intersection_distances_chosen_plume[mask] = intersection_distances[rows,cols].flatten()

    chasing_times = (intersection_distances_chosen_plume/fly_speed)

    arrival_times = release_times+dispersal_travel_times+chasing_times

    return arrival_times[~np.isnan(arrival_times)],chasing_times,rows,cols

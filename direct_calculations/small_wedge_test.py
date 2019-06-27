import extras
from core_functions import f0,f1,f1_wedge,f2,f3,f4,f5
import matplotlib.pyplot as plt
import numpy as np
from pompy import models


wind_mag = 1.
wind_angle = np.pi
cone_angle = np.radians(10.)

#Source locations
location_list = [(200,100) , (300,-100),(-100,-200)]
source_pos = np.array([np.array(tup) for tup in location_list])

#fly headings: = np.pi/3 and 4*np.pi/3
intended_heading_angles = np.radians(np.array([60,110,200,240]))

track_heading_angles,dispersing_speeds = f0(intended_heading_angles,wind_mag,wind_angle)

#Convert track_heading_angles to a list of plume intersection locations for each fly
intersection_distances,dispersal_distances = f1_wedge(
    track_heading_angles,source_pos,wind_angle,cone_angle)

print('intersection_distances: ')
print(np.array2string(intersection_distances))
print('dispersal_distances: ')
print(np.array2string(dispersal_distances))

#Display plumes
im_extents = (-500,500,-500,500)
# gaussianfitPlumes = models.GaussianFitPlume(source_pos,wind_angle,wind_mag)
# conc_im = gaussianfitPlumes.conc_im(im_extents,samples=200)
# conc_im = logisticPlumes.conc_im(im_extents,samples=200)
fig1, ax = plt.subplots()
# plt.imshow(conc_im,extent=im_extents,aspect='equal',origin='lower',cmap='bone_r')
for heading in track_heading_angles:
    plt.plot([0,1000*np.cos(heading)],[0,1000*np.sin(heading)])
for source in source_pos:
    # print(source)
    for offset in [-cone_angle,cone_angle]:
        plt.plot([source[0],source[0]+1000*np.cos(wind_angle+offset)],
            [source[1],source[1]+1000*np.sin(wind_angle+offset)],color='k')
plt.show()

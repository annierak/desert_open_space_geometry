import numpy as np
import matplotlib
import matplotlib.pyplot as plt

wind_angle = np.arctan(4.)
wind_mag = 1.
fly_mag = 1.5
flux_rad = 10.

num_flies = 6000
heading_angles = np.linspace(360./num_flies,360,num_flies)*np.pi/180

hit_angles = np.arctan((wind_mag*np.sin(wind_angle-heading_angles))/fly_mag)+heading_angles
cmap = matplotlib.cm.get_cmap('plasma')
colors = cmap(np.linspace(1./num_flies,1,num_flies))

plt.figure(1)
ax=plt.subplot(1,2,1)
plt.scatter(2*np.cos(heading_angles),2*np.sin(heading_angles),color=colors,alpha=0.01)
plt.scatter(6*np.cos(hit_angles),6*np.sin(hit_angles),color=colors,alpha=0.01)
ax.set_aspect('equal', 'datalim')
ax = plt.subplot(1,2,2,projection='polar')
n,bins,_ = plt.hist(hit_angles,bins=100)
ax.cla()
plt.plot(bins,np.concatenate((n,[n[0]])))
ax.set_yticks([])
plt.show()

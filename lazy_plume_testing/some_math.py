import math
import numpy as np
import matplotlib.pyplot as plt

simulation_time = 30.*60. #seconds
y_locs = np.zeros(2000)

plt.figure()
ax = plt.subplot()
lines, = plt.plot(y_locs,'o',alpha=0.1)
plt.ylim([-100,100])
text = ax.text(0.5,1,' ',transform=ax.transAxes)
text1 = ax.text(0.1,1,' ',transform=ax.transAxes)
p = ax.axhline(y=0)

dt = 0.25

velocity = 1.

times_real_time = 60.
frame_rate = 20
capture_interval = int(np.ceil(times_real_time*(1./frame_rate)/dt))

centre_rel_diff_scale = 2.

centre_rel_diff_scale *= (1./(np.sqrt(dt/0.01)))

plt.ion()
plt.show()

t = 0.

while t<simulation_time:
    for k in range(capture_interval):
        y_locs += dt*np.random.normal(size=np.shape(y_locs))*centre_rel_diff_scale
        lines.set_ydata(y_locs)
        text.set_text(str(t/60)[0:3])
        text1.set_text(np.var(y_locs))
        p.set_ydata(np.std(y_locs))
        plt.pause(0.0001)
        t+=dt
raw_input()

import matplotlib.pyplot as plt
import matplotlib.animation as animate
import matplotlib
matplotlib.use("Agg")
import numpy as np

for i in range(5):
    plt.figure(i)
    ax = plt.subplot(2,1,1)
    plt.plot(np.random.randn(20),np.random.randn(20))
    ax.cla()
    plt.plot(np.random.randn(20),np.random.randn(20))
    ax = plt.subplot(2,1,2)
    plt.plot(np.random.randn(20),np.random.randn(20))

plt.show()

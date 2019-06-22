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

class UpdatingVPatch(object):
    def __init__(self,x_0,width):
        self.rectangle = plt.Rectangle((x_0,0),width,1.,alpha=0.5,color='orange')
    def update(self,new_x_0,new_width):
        self.rectangle.set_x(new_x_0)
        self.rectangle.set_width(new_width)
        # return self.rectangle

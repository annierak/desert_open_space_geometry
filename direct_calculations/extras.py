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

def plot_wedges_old(source_pos,wind_angle,cone_angle):
    length = 1000
    first_arms = np.array(
        (source_pos[:,0]+length*np.cos(wind_angle+cone_angle),
        source_pos[:,1]+length*np.sin(wind_angle+cone_angle))).T
    second_arms = np.array(
        (source_pos[:,0]+length*np.cos(wind_angle-cone_angle),
        source_pos[:,1]+length*np.sin(wind_angle-cone_angle))).T
    #shape traps x 2

    merged = np.stack((first_arms,source_pos,second_arms))
    # print(merged)

    return merged[:,:,0],merged[:,:,1]


def plot_wedges(source_pos,wind_angle,cone_angle):
    length = 2000
    first_arms = np.array(
        (source_pos[:,0]+length*np.cos(wind_angle+cone_angle),
        source_pos[:,1]+length*np.sin(wind_angle+cone_angle))).T
    second_arms = np.array(
        (source_pos[:,0]+length*np.cos(wind_angle-cone_angle),
        source_pos[:,1]+length*np.sin(wind_angle-cone_angle))).T
    #shape traps x 2
    merged = np.stack((first_arms,source_pos,second_arms))
    return merged

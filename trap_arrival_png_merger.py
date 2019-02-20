import sys
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

subfolder = os.getcwd()+'/'+sys.argv[1]
onlyfiles = [f for f in listdir(subfolder) if isfile(join(subfolder, f))]
pngs = [f for f in onlyfiles if f[-4:]=='.png']

#Find the numbers that sort the pngs
number_indices = np.array(list(pngs[0])) == np.array(list(pngs[-1]))
comp_values = []
for png in pngs:
    comp_values.append(
        float(''.join(list(np.array(list(png))[~number_indices]))))

pngs = [pngs[ind] for ind in np.argsort(comp_values)]

os.chdir(subfolder)

images = map(Image.open, pngs)
widths, heights = zip(*(i.size for i in images))

grid_shape = (int(sys.argv[2]),int(sys.argv[3]))

total_width = sum(widths)/grid_shape[0]
total_height = sum(heights)/grid_shape[1]

new_im = Image.new('RGB', (total_width, total_height))

# y_offsets = heights[0]*np.array(range(grid_shape[0]))
y_offset = -1*heights[0]
x_offsets = widths[0]*np.array(range(grid_shape[1]))
x_iter = range(grid_shape[1])
flipper = False
counter = 0
for y in range(grid_shape[0]):
    y_offset += images[counter].size[1]
    if flipper:
        x_iter.reverse()
    for x in x_iter:
        new_im.paste(images[counter], (x_offsets[x],y_offset))
        counter+=1
    flipper = not(flipper)
new_im.save('combo.png')

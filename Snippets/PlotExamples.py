# A simple Plot taken from here:
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/simple_plot.html#sphx-glr-gallery-lines-bars-and-markers-simple-plot-py
# with some tweaks applied to make it look better

# Import libraries:
import matplotlib.pyplot as plt
import numpy as np

# Configure libraries:
plt.style.use('dark_background')
#plt.style.use('default')
#plt.style.use('grayscale')

# Create data for plotting:
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

# Create a plot and show it:
px = 1/plt.rcParams['figure.dpi']  # pixel in inches w
fig, ax = plt.subplots(figsize=(800*px, 400*px))
ax.plot(t, s)
ax.set(xlabel='x-axis', ylabel='y-axis', title='The sine function')
ax.grid()
fig.savefig("test.png") # Will end up in same directory as .py file
plt.show()

# Notes:
#
# Matplotlib supports only to set up the size in inches. To set it up in 
# pixels, one needs to use a conversion factor as demonstrated above. However,
# this doesn't always seem to work properly. For example, when trying to use a 
# size of 600x200, the image ends up being 599x200 instead. There's probably 
# some floating point rounding error and truncation to int going on. That kinda 
# sucks!


#------------------------------------------------------------------------------
#
# See: https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
#
# To define my own style sheet:
# https://matplotlib.org/stable/users/explain/customizing.html#customizing-with-style-sheets
#
#    
# To set up colors:
# https://stackoverflow.com/questions/14088687/how-to-change-plot-background-color
# https://matplotlib.org/stable/gallery/style_sheets/dark_background.html
#
# Some simple example plots:
# https://matplotlib.org/stable/gallery/pyplots/index.html
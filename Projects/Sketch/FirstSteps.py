# A simple Plot taken from here:
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/simple_plot.html#sphx-glr-gallery-lines-bars-and-markers-simple-plot-py    

# Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Configure libraries:
# Set up the style sheet:
plt.style.use('dark_background')
#plt.style.use('default')
#plt.style.use('grayscale')

# See: https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html

# To define my own style sheet:
# https://matplotlib.org/stable/users/explain/customizing.html#customizing-with-style-sheets

    


# Create data for plotting:
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)


# Create a plot and show it:
fig, ax = plt.subplots()
ax.plot(t, s)
ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()
#fig.savefig("test.png") # Saves plot into .png file in same dir as .py file
plt.show()


# To set up colors:
# https://stackoverflow.com/questions/14088687/how-to-change-plot-background-color
# https://matplotlib.org/stable/gallery/style_sheets/dark_background.html

# Some simple example plots:
# https://matplotlib.org/stable/gallery/pyplots/index.html
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, =ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x + i/10.0))
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,

ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=100,
                              init_func=init,
                              interval=20,
                              blit=False)
plt.show()

from mpl_toolkits.mplot3d import Axes3D


def data(i, z, line):
    z=np.sin(x+y+i)
    ax.clear()
    line = ax.plot_surface(x,y,z,color="b")
    return line,


n = 2.*np.pi
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

x = np.linspace(0, n, 100)
y = np.linspace(0, n, 100)
x, y = np.meshgrid(x, y)
z=np.sin(x-y)
line = ax.plot_surface(x, y, z, color='b')

ani = animation.FuncAnimation(fig, data, fargs=(z, line), interval=15, blit=False)

#ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
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


def data(i):

    z=np.sin(np.sqrt(x**2+y**2)+i/10.)/np.sqrt(x**2+y**2)
    ax.clear()
    ax.set_zlim(-3, 3)
    line = ax.plot_surface(x,y,z,cmap=plt.get_cmap('rainbow'))

    return line,


n = 4.*np.pi
fig = plt.figure()

ax = fig.add_subplot(111,projection='3d')

x = np.linspace(-n, n, 100)
y = np.linspace(-n, n, 100)
x, y = np.meshgrid(x, y)
z=np.sin(x-y)
line = ax.plot_surface(x, y, z, color='b')

ani = animation.FuncAnimation(fig=fig, func=data, frames=100, interval=20, blit=False)

ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
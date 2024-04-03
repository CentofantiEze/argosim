import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plot_beam(beam, pRng = (-0.1, 0.5), ax=None, fig=None):
    # Imshow min and max values.
    zMin = np.nanmin(beam)
    zMax = np.nanmax(beam)
    zRng = zMin - zMax
    zMin -= zRng * pRng[0]
    zMax += zRng * pRng[1]
    if ax==None or fig==None:
        fig, ax = plt.subplots(1,1)
    im = ax.imshow(np.fft.ifftshift(beam), vmin=zMin, vmax=zMax)
    fig.colorbar(im, ax=ax)
    if ax==None or fig==None:
        plt.show()

def plot_antenna_arr(array, ax=None, fig=None):
    if ax==None or fig==None:
        fig, ax = plt.subplots(1,1)
    ax.scatter(array[:,0], array[:,1],s=20, c='gray')
    for i, txt in enumerate(range(1,len(array)+1,1)):
        ax.annotate(txt, (array[i,0], array[i,1]))
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        x_lim=max(abs(array[:,0]))*1.1
        y_lim=max(abs(array[:,1]))*1.1
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_aspect('equal', adjustable='box')
    if ax==None or fig==None:
        plt.show()

def plot_baselines(visibilities, n_baselines=None, ax=None, fig=None):
    if ax==None or fig==None:
        fig, ax = plt.subplots(1,1)
    ax.scatter(visibilities[:,0], visibilities[:,1],s=0.4, c='gray')
    if n_baselines is not None:
        delta = int(visibilities.shape[0]/2)
        ax.scatter(visibilities[delta:delta+n_baselines,0], visibilities[delta:delta+n_baselines,1], s=2,c='k')
    ax.set_xlabel(r'u x $\lambda$ [m]')
    ax.set_ylabel(r'v x $\lambda$ [m]')
    ax.set_xlim([np.min(visibilities), np.max(visibilities)])
    ax.set_ylim([np.min(visibilities), np.max(visibilities)])
    ax.set_aspect('equal', adjustable='box')
    if ax==None or fig==None:
        plt.show()

def plot_sky(image):
    plt.imshow(image)
    plt.show()
    print('Image shape:', image.shape)
    print('Image range: ({},{})'.format(np.min(image), np.max(image)))

def plot_sky_uv(sky_uv):
    plt.imshow(np.abs(np.fft.fftshift(sky_uv)), norm=matplotlib.colors.LogNorm())
    plt.show()

def plot_sampled_sky(sky_uv):
    plt.imshow(np.abs(sky_uv)+1e-3, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.show()
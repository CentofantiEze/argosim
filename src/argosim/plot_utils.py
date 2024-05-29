"""Plot utils.

This module contains functions to plot the antenna array, beam, baselines, 
uv-coverage and sky models.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_beam(beam, pRng = (-0.1, 0.5), ax=None, fig=None):
    """Plot beam.
    
    Function to plot the synthesised beam image.
    
    Parameters
    ----------
    beam : np.ndarray
        The beam image.
    pRng : tuple
        The range of the colorbar.
    ax : matplotlib.axes.Axes
        The axis to plot the beam. For plotting on a specific subplot axis.
    fig : matplotlib.figure.Figure
        The figure to plot the beam. For plotting on a specific subplot axis.
        
    Returns
    -------
    None
    """
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
    """Plot antenna array.

    Function to plot the antenna array in ground coordinates.

    Parameters
    ----------
    array : np.ndarray
        The antenna array positions in the ground.
    ax : matplotlib.axes.Axes
        The axis to plot the antenna array. For plotting on a specific subplot axis.
    fig : matplotlib.figure.Figure
        The figure to plot the antenna array. For plotting on a specific subplot axis.
    
    Returns
    -------
    None
    """
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
    """Plot baselines.

    Function to plot the baselines in uv-space.

    Parameters
    ----------
    visibilities : np.ndarray
        The visibilities baselines in uv-space.
    n_baselines : int
        The number of baselines. Only for aperture synthesis. 
        Plot the baselines halfway through the observations.
    ax : matplotlib.axes.Axes
        The axis to plot the baselines. For plotting on a specific subplot axis.
    fig : matplotlib.figure.Figure
        The figure to plot the baselines. For plotting on a specific subplot axis.

    Returns
    -------
    None
    """
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
    """Plot sky.

    Function to plot the sky model.

    Parameters
    ----------
    image : np.ndarray
        The sky model image.

    Returns
    -------
    None
    """
    plt.imshow(image)
    plt.show()
    print('Image shape:', image.shape)
    print('Image range: ({},{})'.format(np.min(image), np.max(image)))

def plot_sky_uv(sky_uv):
    """Plot sky uv.

    Function to plot the sky model in uv-space in logarithmic amplitud sale.

    Parameters
    ----------
    sky_uv : np.ndarray
        The sky model in uv-space.The inverse fourier transform of the sky model.
    
    Returns
    -------
    None
    """
    plt.imshow(np.abs(np.fft.fftshift(sky_uv)), norm=matplotlib.colors.LogNorm())
    plt.show()

def plot_sampled_sky(sky_uv):
    """Plot sampled sky.

    Function to plot the sky model samples in uv-space according to the uv-coverage.

    Parameters
    ----------
    sky_uv : np.ndarray
        The sampled sky model in uv-space.

    Returns
    -------
    None
    """
    plt.imshow(np.abs(sky_uv)+1e-3, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.show()
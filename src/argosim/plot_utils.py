"""Plot utils.

This module contains functions to plot the antenna array, beam, baselines, 
uv-coverage and sky models.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_beam(beam, pRng=(-0.1, 0.5), ax=None, fig=None):
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
    if ax == None or fig == None:
        fig, ax = plt.subplots(1, 1)
    im = ax.imshow(np.fft.ifftshift(beam), vmin=zMin, vmax=zMax)
    fig.colorbar(im, ax=ax)
    if ax == None or fig == None:
        plt.show()


def plot_antenna_arr(array, ax=None, fig=None, title="Array"):
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
    title : str
        The title of the plot.

    Returns
    -------
    None
    """
    # Center the array
    array = array - np.mean(array, axis=0)
    if ax == None or fig == None:
        fig, ax = plt.subplots(1, 1)
    ax.scatter(array[:, 0], array[:, 1], s=20, c="gray")
    for i, txt in enumerate(range(1, len(array) + 1, 1)):
        ax.annotate(txt, (array[i, 0], array[i, 1]))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        x_lim = np.max(np.abs(array)) * 1.1
        y_lim = x_lim
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    if ax == None or fig == None:
        plt.show()


def plot_baselines(visibilities, ax=None, fig=None, ENU=False, title="Baselines"):
    """Plot baselines.

    Function to plot the baselines in uv-space.

    Parameters
    ----------
    visibilities : np.ndarray
        The visibilities baselines in uv-space.
    ax : matplotlib.axes.Axes
        The axis to plot the baselines. For plotting on a specific subplot axis.
    fig : matplotlib.figure.Figure
        The figure to plot the baselines. For plotting on a specific subplot axis.
    ENU : bool
        If True, plot the baselines in East-North-Up coordinates. Otherwise, plot in uv-space.
    title : str
        The title of the plot.

    Returns
    -------
    None
    """
    if ax == None or fig == None:
        fig, ax = plt.subplots(1, 1)
    ax.scatter(visibilities[:, 0], visibilities[:, 1], s=0.4, c="gray")
    # if n_baselines is not None:
    #     delta = int(visibilities.shape[0]/2)
    #     ax.scatter(visibilities[delta:delta+n_baselines,0], visibilities[delta:delta+n_baselines,1], s=2,c='k')
    if ENU:
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
    else:
        ax.set_xlabel(r"u x $\lambda$ [m]")
        ax.set_ylabel(r"v x $\lambda$ [m]")
    ax.set_xlim([np.min(visibilities), np.max(visibilities)])
    ax.set_ylim([np.min(visibilities), np.max(visibilities)])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    if ax == None or fig == None:
        plt.show()


def plot_sky(image, fov_size=(1.0, 1.0), ax=None, fig=None, title="Sky"):
    """Plot sky.

    Function to plot the sky model.

    Parameters
    ----------
    image : np.ndarray
        The sky model image.
    fov_size : tuple
        The field of view size in degrees.

    Returns
    -------
    None
    """
    if ax == None or fig == None:
        fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        image,
        extent=[-fov_size[0] / 2, fov_size[0] / 2, -fov_size[1] / 2, fov_size[1] / 2],
    )
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x [deg]")
    ax.set_ylabel("y [deg]")
    ax.set_title("{} ({}x{})".format(title, image.shape[0], image.shape[1]))
    if ax == None or fig == None:
        plt.show()


def plot_sky_uv(sky_uv, fov_size):
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
    max_u = (180 / np.pi) * sky_uv.shape[0] / (2 * fov_size[0])
    max_v = (180 / np.pi) * sky_uv.shape[1] / (2 * fov_size[1])

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(np.abs(sky_uv), extent=[-max_u, max_u, -max_v, max_v])
    plt.xlabel(r"$u\times\lambda$ [m]")
    plt.ylabel(r"$v\times\lambda$ [m]")
    plt.title("Amplitude")
    plt.subplot(122)
    plt.imshow(np.angle(sky_uv), extent=[-max_u, max_u, -max_v, max_v])
    plt.xlabel(r"$u\times\lambda$ [m]")
    plt.ylabel(r"$v\times\lambda$ [m]")
    plt.title("Phase")
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
    plt.imshow(np.abs(sky_uv) + 1e-3, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.show()

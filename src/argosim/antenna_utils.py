"""Antenna utils.

This module contains functions to generate antenna arrays, compute its baselines, 
perform aperture synthesis, obtain uv-coverage and get observations from sky models.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy.random as rnd
from PIL import Image

########################################
#      Generate antenna positions      #
########################################

def random_antenna_pos(E_lim = 1000, N_lim =1000, U_lim=0):
    """Random antenna pos.
    
    Function to generate a random antenna location in ENU coordinates.
    Antenna lies in the range:
        [-E_lims/2, E_lims/2]x[-N_lims/2, N_lims/2]x[0, U_lims].
    
    Parameters
    ----------
    E_lim : int
        The east coordinate span width of the antenna position in meters.
    N_lim : int
        The north coordinate span width of the antenna position in meters.
    U_lim : int
        The up coordinate span width of the antenna position in meters.
        
    Returns
    -------
    antenna_pos : np.ndarray
        The antenna position in ENU coordinates.
    """
    # Return (x,y) random location for single dish
    return rnd.random_sample(3)*np.array([E_lim,N_lim,U_lim]) - np.array([E_lim, N_lim, 0.])/2

def circular_antenna_arr(n_antenna= 3, r=300):
    """Circular antenna arr.

    Function to generate a circular antenna array. Antennas lie in a circumference 
    of radius 'r' from the center [0,0] and are equally spaced.

    Parameters
    ----------
    n_antenna : int
        The number of antennas in the array. 
    r : int
        The radius of the antenna array in meters.

    Returns
    -------
    antenna_arr : np.ndarray
        The antenna array positions in ENU coordinates.
    """
    # Return list of 'n' antenna locations (x_i, y_i) equally spaced over a 'r' radius circumference.
    return np.array([[np.cos(angle)*r, np.sin(angle)*r, 0.] for angle in [2*np.pi/n_antenna*i for i in range(n_antenna)]])

def y_antenna_arr(n_antenna=5, r=500, alpha=0):
    """Y antenna arr.

    Function to generate a Y-shaped antenna array. Antennas lie equispaced in three radial arms
    of 120 degrees each.

    Parameters
    ----------
    n_antenna : int
        The number of antennas per arm.
    r : int
        The radius of the antenna array in meters.
    alpha : int
        The angle of the first arm.

    Returns
    -------
    antenna_arr : np.ndarray
        The antenna array positions in ENU coordinates.
    """
    # Return list of 'n' antenna locations (x_i, y_i) equispaced on three (120 deg) radial arms.
    step = r/n_antenna
    return np.array([ [np.array([(i+1)*step*np.cos(angle/180*np.pi), (i+1)*step*np.sin(angle/180*np.pi), 0.]) for i in range(n_antenna)] for angle in [alpha, alpha+120, alpha+240] ]).reshape((3*n_antenna,3))

def random_antenna_arr(n_antenna=3, E_lim=1000, N_lim=1000, U_lim=0):
    """Random antenna arr.

    Function to generate a random antenna array. Antennas lie randomly distributed
    in the range:
        [-E_lims/2, E_lims/2]x[-N_lims/2, N_lims/2]x[0, U_lims].

    Parameters
    ----------
    n_antenna : int
        The number of antennas in the array.
    E_lim : int
        The east coordinate span width of the antenna positions in meters.
    N_lim : int
        The north coordinate span width of the antenna positions in meters.
    U_lim : int
        The up coordinate span width of the antenna positions in meters.

    Returns
    -------
    antenna_arr : np.ndarray
        The antenna array positions in ENU coordinates.
    """
    # Return list of 'n' antenna locations (x_i, y_i) randomly distributed.
    return np.array([random_antenna_pos(E_lim, N_lim, U_lim) for i in range(n_antenna)])

########################################
#  Compute baselines and uv sampling   #
########################################

def get_baselines(array):
    """Get baselines.

    Function to compute the baselines of an antenna array.

    Parameters
    ----------
    array : np.ndarray
        The antenna array positions.

    Returns
    -------
    baselines : np.ndarray
        The baselines of the antenna array.
    """
    # Get the baseline for every combination of antennas i-j.
    # Remove the i=j baselines: np.delete(array, list, axis=0) -> delete the rows listed on 'list' from array 'array'. 
    return np.delete(np.array([antenna_i-antenna_j for antenna_i in array for antenna_j in array]), [(len(array)+1)*n for n in range(len(array))], 0)

def ENU_to_XYZ(b_ENU, lat = 35./180*np.pi):
    """ENU to XYZ.

    Function to convert the baselines from East-North-Up (ENU) to XYZ coordinates.

    Parameters
    ----------
    b_ENU : np.ndarray
        The baselines in ENU coordinates.
    lat : float
        The latitude of the antenna array in radians.

    Returns
    -------
    X : np.ndarray
        The X coordinate of the baselines in XYZ coordinates.
    Y : np.ndarray
        The Y coordinate of the baselines in XYZ coordinates.
    Z : np.ndarray
        The Z coordinate of the baselines in XYZ coordinates.
    """
    # Compute baseline length, Azimuth and Elevation angles
    D = np.sqrt(np.sum(b_ENU**2, axis=1))
    A = np.arctan2(b_ENU[:,0], b_ENU[:,1])
    E = np.arcsin(b_ENU[:,2]/D)
    # Compute the baseline in XYZ coordinates
    X = D*(np.cos(lat)*np.sin(E)-np.sin(lat)*np.cos(E)*np.cos(A))
    Y = D*np.cos(E)*np.sin(A)
    Z = D*(np.sin(lat)*np.sin(E)+np.cos(lat)*np.cos(E)*np.cos(A))

    return X,Y,Z

def XYZ_to_uvw(X, Y, Z, dec=30./180*np.pi , ha=0., f=1420e6):
    """XYZ to uvw.

    Get the uvw sampling points from the XYZ coordinates given a 
        source declination, hour angle and frequency.

    Parameters
    ----------
    X : np.ndarray
        The X coordinate of the baselines in XYZ coordinates.
    Y : np.ndarray
        The Y coordinate of the baselines in XYZ coordinates.
    Z : np.ndarray
        The Z coordinate of the baselines in XYZ coordinates.
    dec : float
        The declination of the source in radians.
    ha : float
        The hour angle of the source in radians.
    f : float
        The frequency of the observation in Hz.

    Returns
    -------
    u : np.ndarray
        The u coordinate of the baselines in uvw coordinates.
    v : np.ndarray
        The v coordinate of the baselines in uvw coordinates.
    w : np.ndarray
        The w coordinate of the baselines in uvw coordinates.
    """
    c = 299792458
    lam_inv = f/c
    u = lam_inv*(np.sin(ha)*X+np.cos(ha)*Y)
    v = lam_inv*(-np.sin(dec)*np.cos(ha)*X+np.sin(dec)*np.sin(ha)*Y+np.cos(dec)*Z)
    w = lam_inv*(np.cos(dec)*np.cos(ha)*X-np.cos(dec)*np.sin(ha)*Y+np.sin(dec)*Z)
    return u,v,w
    

def uv_track_multiband(b_ENU, lat = 35./180*np.pi, dec=35./180*np.pi, track_time=0., t_0=0., n_times = 1 , f=1420e6, df=0., n_freqs=1):
    """Uv track multiband.

    Function to compute the uv sampling baselines for a given observation time and frequency range.

    Parameters
    ----------
    b_ENU : np.ndarray
        The baselines in ENU coordinates.
    lat : float
        The latitude of the antenna array in radians.
    dec : float
        The declination of the source in radians.
    track_time : float
        The duration of the tracking in hours.
    t_0 : float
        The initial tracking time in hours.
    n_times : int
        The number of time steps.
    f : float
        The central frequency of the observation in Hz.
    df : float
        The frequency range of the observation in Hz.
    n_freqs : int
        The number of frequency samples.

    Returns
    -------
    track : np.ndarray
        The uv sampling baselines listed for each time step and frequency.
    """
    # Compute the baselines in XYZ coordinates
    X, Y, Z = ENU_to_XYZ(b_ENU, lat)
    # Compute the time steps
    h = np.linspace(t_0, t_0+track_time, n_times)*np.pi/12
    # Compute the frequency range
    f_range = np.linspace(f-df/2, f+df/2, n_freqs)

    track = []
    for t in h:
        multi_band = []
        for f_ in f_range:
            u,v,w = XYZ_to_uvw(X, Y, Z, dec, t, f_)
            multi_band.append(np.array([u,v,w]))
        track.append(multi_band)
    track = np.array(track).swapaxes(-1,-2).reshape(-1,3)

    return track

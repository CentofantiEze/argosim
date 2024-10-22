"""Data uils. 

This module contains functions to generate synthetic observations for the Argosim project.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import numpy as np


def gauss_source(
        nx=512,
        ny=512, 
        mu=np.array([0,0]), 
        sigma=np.eye(2), 
        fwhm_pix = 64
    ):
    """Gauss source.

    Function to generate a 2D Gaussian source.

    Parameters
    ----------
    nx : int
        The output image first dimension.
    ny : int
        The output image second dimension.
    mu : np.ndarray
        The mean of the Gaussian source in the [-1,1]x[-1,1] range.
    sigma : np.ndarray
        The covariance matrix of the Gaussian source.
    fwhm_pix : float
        The FWHM of the Gaussian source in pixels.

    Returns
    -------
    source : np.ndarray
        Image of size (nx,ny) containing the 2D Gaussian source.
    """
    fwhm = 2.355
    x, y = np.meshgrid(np.linspace(-fwhm/2, fwhm/2, nx), np.linspace(-fwhm/2, fwhm/2, ny))

    sigma = sigma/(nx*ny)*fwhm_pix**2/np.sqrt(np.linalg.det(sigma))

    X_unroll = np.array([x.reshape(-1)-mu[0], y.reshape(-1)-mu[1]])
    sigminv = np.linalg.inv(sigma)
    sigminv.dot(X_unroll).shape
    Q = np.sum(np.multiply(X_unroll,sigminv.dot(X_unroll)),axis=0).reshape(nx,ny)
    return np.exp(-Q/2)#/(np.sqrt(2*np.pi*np.abs(np.linalg.det(sigma))))

def sigma2d(min_var = 5, cov_lim = 0.5):
    """Sigma 2D.

    Function to generate a random 2D covariance matrix.

    Parameters
    ----------
    min_var : float
        The minimum variance of both gaussian components.
    cov_lim : float
        The limit of the covariance between gaussian components.

    Returns
    -------
    sigma : np.ndarray
        The 2D covariance matrix.
    """
    var_1 = np.random.rand()+min_var
    # Limit eccentricity
    var_2 = np.random.rand()+min_var
    # Cov <= sqrt(var1 x var2)
    cov12 = (np.random.rand()*2-1)*np.sqrt(var_1*var_2)*cov_lim
    return np.array([[var_1, cov12],[cov12, var_2]])

def mu2d():
    """Mu 2D.

    Function to generate a random 2D mean vector in the range [-1,1]x[-1,1].

    Returns
    -------
    mu : np.ndarray
        The 2D mean vector.
    """
    return np.random.rand(2)*2-1

def random_source(shape,pix_size):
    """Random source.

    Function to generate 2D Gaussian source with random mean and covariance.

    Parameters
    ----------
    shape : tuple
        The output image shape.
    pix_size : float
        The size in pixels of the Gaussian source.
    
    Returns
    -------
    source : np.ndarray
        Image of size (nx,ny) containing the 2D Gaussian source.
    """
    mu = mu2d()
    sigma = sigma2d()
    return gauss_source(shape[0], shape[1], mu, sigma, pix_size)

def n_source_sky(shape_px, pix_size_list, source_intensity_list):
    """N source sky.

    Function to generate a sky image with multiple Gaussian sources at random positions.

    Parameters
    ----------
    shape_px : tuple
        The image size in pixels (Nx, Ny).
    pix_size_list : list
        The size in pixels of the Gaussian sources.
    source_intensity_list : list
        The intensity of each Gaussian source in the final image. 
        The sum of all the sources should be equal to 1 to have a normalized image. 

    Returns
    -------
    sky : np.ndarray
        Image of size (nx,ny) containing the sky model.
    """
    return sum([random_source((shape_px[0],shape_px[1]), pix_size)*intensity for pix_size, intensity in zip(pix_size_list,source_intensity_list)])

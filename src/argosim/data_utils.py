import numpy as np

def gauss_source(nx=512,ny=512, mu=np.array([0,0]), sigma=np.eye(2), fwhm_pix = 64):
    fwhm = 2.355
    x, y = np.meshgrid(np.linspace(-fwhm/2, fwhm/2, nx), np.linspace(-fwhm/2, fwhm/2, ny))

    sigma = sigma/(nx*ny)*fwhm_pix**2/np.sqrt(np.linalg.det(sigma))

    X_unroll = np.array([x.reshape(-1)-mu[0], y.reshape(-1)-mu[1]])
    sigminv = np.linalg.inv(sigma)
    sigminv.dot(X_unroll).shape
    Q = np.sum(np.multiply(X_unroll,sigminv.dot(X_unroll)),axis=0).reshape(nx,ny)
    return np.exp(-Q/2)#/(np.sqrt(2*np.pi*np.abs(np.linalg.det(sigma))))

def sigma2d(min_var = 5, cov_lim = 0.5):
    var_1 = np.random.rand()+min_var
    # Limit eccentricity
    var_2 = np.random.rand()+min_var
    # Cov <= sqrt(var1 x var2)
    cov12 = (np.random.rand()*2-1)*np.sqrt(var_1*var_2)*cov_lim
    return np.array([[var_1, cov12],[cov12, var_2]])

def mu2d():
    return np.random.rand(2)*2-1

def random_source(shape,pix_size):
    mu = mu2d()
    sigma = sigma2d()
    return gauss_source(shape[0], shape[1], mu, sigma, pix_size)

def n_source_sky(shape, pix_size_list, source_intensity_list):
    return sum([random_source((shape[0],shape[1]), pix_size)*intensity for pix_size, intensity in zip(pix_size_list,source_intensity_list)])

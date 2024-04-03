import numpy as np
import matplotlib.pyplot as plt
import argosim

from argosim.antenna_utils import *
from argosim.plot_utils import *
from argosim.data_utils import *

plt.rcParams['image.cmap'] = 'afmhot'

# Antenna parameters
x_lim = 30000
y_lim = 30000
n_antenna = 3*9
n_baselines = n_antenna*(n_antenna-1)
radius = 30000
alpha = 120

# Source tracking parameters
source_decl = -20/180*np.pi
array_lat = 19.1/180*np.pi
track_time = 4
delta_t = 5/60 # 300 segs
t_0 = -0.5

# Sky model
nx = 512
ny = 512
pix_sizes = [10, 7, 10, 5]
amplitudes = [0.25, 0.25, 0.25, 0.25]
sky = n_source_sky((nx,ny), pix_sizes, amplitudes)

# UV parameters
uv_dim = sky.shape[0]

arr = y_antenna_arr(int(n_antenna/3), radius, alpha)
track = uv_time_int(arr, array_lat, source_decl, track_time, delta_t, t_0)
# Get uv mask
uv_plane = get_uv_plane(track,uv_dim)
uv_plane_mask = get_uv_mask(uv_plane)
# Get the dirty beam
beam = get_beam(uv_plane_mask)
# Get the sky model FT
sky_uv = get_sky_uv(sky)
# Sample Fourier space
obs_uv = get_obs_uv(sky_uv,uv_plane_mask)
# Get observed sky
obs_sky = get_obs_sky(obs_uv, abs=True)

fig, ax = plt.subplots(1,3,figsize=(20,5))
im=ax[0].imshow(sky)
plt.colorbar(im, ax=ax[0])
plot_beam(beam, ax=ax[1],fig=fig)
im=ax[2].imshow(obs_sky)
plt.colorbar(im, ax=ax[2])

ax[0].set_title('Sky model')
ax[1].set_title('Dirty beam')
ax[2].set_title('Observation')

plt.savefig('/home/figures/observation.pdf')
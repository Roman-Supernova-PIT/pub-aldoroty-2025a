import galsim
import numpy as np
from roman_imsim.utils import roman_utils
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import pickle as pkl
from phrosty.utils import get_roman_bands
from phrosty import plotaesthetics
from phrosty.plotting import roman_sca_plot
from astropy.nddata import Cutout2D
from itertools import combinations

scas = np.arange(1, 19, 1)
imgdetails = {'band': get_roman_bands(), 
              'pointing': [35083, 35138, 35193, 35248, 35303, 35358, 35413]
             }
x, y = [2044., 2044.]
galsim_psf_paths = [f'example_galsim_psfs/testpsf_{b}_{s}.fits' for b in imgdetails['band'] for s in scas]

def alpha_func(x, y):
    return abs(x-y)/(x+y)

asymmetry_indices = {band: [] for band in get_roman_bands()}
ellipticity = {f'{band}_{e}': [] for band in get_roman_bands() for e in ['e1', 'e2']}

def get_imsim_psf(x, y, band, pointing, sca, size=201, config_yaml_file=None,
                  psf_path=None, force=False, logger=None, **kwargs):

    """
    Retrieve the PSF from roman_imsim/galsim, and transform the WCS so that CRPIX and CRVAL
    are centered on the image instead of at the corner.

    kwargs match getPSF_Image args, listed here with their defaults:
    pupil_bin=8,
    sed=None,
    oversampling_factor=1,
    include_photonOps=False,
    n_phot=1e6

    force parameter does not currently do anything.
    """

    if psf_path is None:
        raise ValueError( "psf_path can't be None" )

    # Get PSF at specified ra, dec.
    assert config_yaml_file is not None, "config_yaml_file is a required argument"
    config_path = config_yaml_file
    config = roman_utils(config_path,pointing,sca)
    psf = config.getPSF_Image(size,x,y,**kwargs)
    psf.write( str(psf_path) )

for band, pointing in zip(imgdetails['band'], imgdetails['pointing']):
    imgarrays = []
    for sca in scas:
        path = f'example_galsim_psfs/{band}_{sca}.fits'
        get_imsim_psf(x, y, band, pointing, sca, 
                  include_photonOps = True,
                  force = True,
                  config_yaml_file='/sn_info_dir/tds.yaml',
                  oversampling_factor=3,
                  include_pixel = True,
                  psf_path=path)

        hdu = fits.open(path)
        img = hdu[0].data
        imgarrays.append(img[883:926,883:926])
    imgarrays = np.array(imgarrays)    
    fig = roman_sca_plot(imgarrays, scas, show_sca_id = False, return_fig = True,
                         cmap = 'viridis', residual_plot = False, title = band,
                         savefig = True, vlims = [0, 0.01], savepath = f'figs/{band}_sca_psf.png')

    for sca in scas:
        # Get the asymmetry index for all SCAs in all bands at the center of each SCA. 
        path = f'example_galsim_psfs/{band}_{sca}.fits'
        
        hdu = fits.open(path)
        img = hdu[0].data
        cutout_size = (np.array((img.shape))/2)[0]
        init_cutout_center = (np.array((img.shape))/4)[0]
        cutout_centers = np.array([init_cutout_center, init_cutout_center + cutout_size])
        
        cutout_sums = []
        for i in [0, 1]:
            for j in [0, 1]:
                xy = (cutout_centers[i], cutout_centers[j])
                cutout = Cutout2D(img, xy, cutout_size, limit_rounding_method=np.round)
                cutout_sums.append(np.nansum(cutout.data))
      
        sum_alph = 0
        iterable = list(combinations(cutout_sums,2))
        for pair in iterable:
            alph = alpha_func(*pair)
            sum_alph += alph

        ai = sum_alph/4
        asymmetry_indices[band].append(ai)

        # Get ellipticity adaptive moments
        gimg = galsim.Image(img)
        shape = galsim.hsm.FindAdaptiveMom(gimg, strict=False)
        ellipticity[f'{band}_e1'].append(shape.observed_e1)
        ellipticity[f'{band}_e2'].append(shape.observed_e2)

ai_tab = Table(asymmetry_indices)
ai_tab['sca'] = np.arange(1, 19, 1)
ai_tab = ai_tab['sca', *get_roman_bands()]
for col in ai_tab.itercols():
    if col.info.dtype.kind == 'f':        
        np.around(col, decimals=4, out=col)
ai_tab.write('asymmetry_index.csv', format='csv', overwrite=True)

e_tab = Table(ellipticity)
e_tab['sca'] = np.arange(1,19,1)
e_tab.write('ellipticity.csv', format='csv', overwrite=True)


        

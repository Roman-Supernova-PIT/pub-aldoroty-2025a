# This is intended to be run in Rob's container on Perlmutter.
# See: https://github.com/Roman-Supernova-PIT/phrosty/tree/main/examples/perlmutter

# IMPORTS Standard:
from multiprocessing import Pool
import os
import pickle as pkl
import numpy as np
from itertools import product
import argparse

# IMPORTS Astro:
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.wcs import WCS
from photutils.background import Background2D
from galsim import roman

# IMPORTS Internal:
from phrosty.photometry import ap_phot, psf_phot
from phrosty.utils import get_exptime
from phrosty.utils import read_truth_txt

class Image:
    """
    This carries information about the grid on which the PSF
    is being calculated. It's information that applies to the
    entire chip: grid edges, grid centers.

    """

    def __init__(self, band, pointing, sca, gridsize, N_stack):

        print('Making image object for', band, pointing, sca)

        self.band = band
        self.pointing = pointing
        self.sca = sca
        self.imgpath = f'/sims_dir/RomanTDS/images/simple_model/{band}/{pointing}/Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits.gz'

        with fits.open(self.imgpath) as hdu:
            self.img_ = hdu[1].data
            self.wcs = WCS(hdu[0].header)
        bkg = Background2D(self.img_,box_size=(73,73))
        self.img = self.img_ - bkg.background

        self.N_stack = N_stack
        self.gridsize = gridsize
        self.cutoutsize = int(4088/self.gridsize)
        self.grid_edges = np.linspace(self.cutoutsize, 4088, self.gridsize)
        self.grid_centers = np.linspace(0.5 * self.cutoutsize, 4088 - 0.5 * self.cutoutsize, gridsize)

        self.zpt_imagelist = self.get_zpt_imagelist()
        self.zpt_images = self.get_zpt_images()

        print('Done making image object.')

    def get_zpt_imagelist(self):
        # Get a table that doesn't include the image you're working on. 
        obseq_data_path = '/sims_dir/RomanTDS/Roman_TDS_obseq_11_6_23.fits'
        obseq_data = Table.read(obseq_data_path, format='fits')
        obseq_data['pointing'] = np.arange(0, len(obseq_data), 1)
        obseq_data = obseq_data[obseq_data['filter'] == self.band]
        obseq_data = obseq_data[obseq_data['pointing'] != self.pointing]
        zpt_imagelist = obseq_data[:self.N_stack]

        return zpt_imagelist

    def get_zpt_images(self):
        """
        Makes a list of all the images involved in ZPT calculation
        and background subtracts them. 

        """
        zpt_images = []
        for row in self.zpt_imagelist:
            imgpath = f'/sims_dir/RomanTDS/images/simple_model/{self.band}/{row["pointing"]}/Roman_TDS_simple_model_{self.band}_{row["pointing"]}_{self.sca}.fits.gz'
            with fits.open(imgpath) as hdu:
                zptimg_ = hdu[1].data

            bkg = Background2D(zptimg_.data,box_size=(73,73))
            zptimg = zptimg_.data - bkg.background
            zpt_images.append(zptimg)

        return zpt_images

class Quad(Image):

    """
    Contains information about the particular quadrant/grid element
    you're working in. 
    
    """

    def __init__(self, imageclass, i, j, zpt_min, zpt_max, forced_phot):

        self.band = imageclass.band
        self.pointing = imageclass.pointing
        self.sca = imageclass.sca
        self.cutoutsize = imageclass.cutoutsize
        self.img = imageclass.img

        self.quad_edges_x = [imageclass.grid_edges[i] - imageclass.cutoutsize, imageclass.grid_edges[i]]
        self.quad_edges_y = [imageclass.grid_edges[j] - imageclass.cutoutsize, imageclass.grid_edges[j]]

        self.quad_cen_x = imageclass.grid_centers[i]
        self.quad_cen_y = imageclass.grid_centers[j]

        self.stars = self.get_stars()

        self.zpt_imagelist = imageclass.zpt_imagelist
        self.zpt_images = imageclass.zpt_images

        self.zpt_min = zpt_min
        self.zpt_max = zpt_max

        self.zpt_stars = self.get_zpt_stars()
        self.zpt_idx = self.zpt_idx()

        self.psf_model = self.get_psf()
        self.forced_phot = forced_phot

        self.ap_zpt, self.psf_zpt = self.calculate_zpts()

    def get_zpt_stars(self):
        zpt_stars = []
        for row in self.zpt_imagelist:
            truthtab = read_truth_txt(path = None, band = self.band, pointing = row['pointing'], sca = self.sca)
            z_stars = truthtab[truthtab['obj_type'] == 'star']
            z_stars = z_stars[z_stars['x'] < self.quad_edges_x[1]]
            z_stars = z_stars[z_stars['x'] > self.quad_edges_x[0]]
            z_stars = z_stars[z_stars['y'] < self.quad_edges_y[1]]
            z_stars = z_stars[z_stars['y'] > self.quad_edges_y[0]]

            exptime = get_exptime(self.band)
            area_eff = roman.collecting_area
            galsim_zp = roman.getBandpasses()[self.band].zeropoint
            truthmag = -2.5*np.log10(z_stars['flux']) + 2.5*np.log10(exptime * area_eff) + galsim_zp
            z_stars['truth_mag'] = truthmag

            zpt_stars.append(z_stars)

        return zpt_stars

    def zpt_idx(self):

        exptime = get_exptime(self.band)
        area_eff = roman.collecting_area
        galsim_zp = roman.getBandpasses()[self.band].zeropoint
        truthmag = -2.5 * np.log10(vstack(self.zpt_stars)['flux']) + 2.5*np.log10(exptime * area_eff) + galsim_zp
        zpt_idx = np.where((truthmag < self.zpt_max) & (truthmag > self.zpt_min))[0]

        return zpt_idx

    def get_psf(self):
        # psfpath = f'./gridded_psfs/{self.band}/{self.sca}/{self.cutoutsize}_{self.quad_cen_x}_{self.quad_cen_y}_-_{self.zpt_min}_{self.zpt_max}_-_{self.band}_{self.sca}.psf'
        psfpath = f'/home/paper_analysis/gridded_psfs/{self.band}/{self.sca}/{self.cutoutsize}_{self.quad_cen_x}_{self.quad_cen_y}_-_{self.zpt_min}_{self.zpt_max}_-_{self.band}_{self.sca}.psf'
        with open(psfpath, 'rb') as f:
            psf = pkl.load(f)

        return psf['psf']

    def calculate_zpts(self):

        # Aperture photometry first.
        zpt_apphot = []
        zpt_images = []
        for img, starlist in zip(self.zpt_images, self.zpt_stars):
            if len(starlist) > 0:
                ap = ap_phot(img, starlist)
                ap['xcentroid'] -= 1
                ap['ycentroid'] -= 1
                ap['xcentroid'].name = 'x'
                ap['ycentroid'].name = 'y'
                zpt_apphot.append(ap)
                zpt_images.append(img)

        zpt_ap_fitmag = -2.5 * np.log10(vstack(zpt_apphot)['aperture_sum'])
        ap_zpt = np.nanmedian(vstack(self.zpt_stars)['truth_mag'] - zpt_ap_fitmag)

        # Then PSF photometry zero point. 
        zpt_psfphot = []
        for img, init_params in zip(zpt_images, zpt_apphot):
            if len(starlist) > 0:
                psfresults = psf_phot(img, self.psf_model, init_params, forced_phot = self.forced_phot)
                zpt_psfphot.append(psfresults)

        zpt_psf_fitmag = - 2.5 * np.log10(vstack(zpt_psfphot)['flux_fit'])
        psf_zpt = np.nanmedian(vstack(self.zpt_stars)['truth_mag'] - zpt_psf_fitmag)

        return ap_zpt, psf_zpt


    def get_stars(self):
        truthtab = read_truth_txt(path = None, band = self.band, pointing = self.pointing, sca = self.sca)
        stars = truthtab[truthtab['obj_type'] == 'star']
        stars = stars[stars['x'] < self.quad_edges_x[1]]
        stars = stars[stars['x'] > self.quad_edges_x[0]]
        stars = stars[stars['y'] < self.quad_edges_y[1]]
        stars = stars[stars['y'] > self.quad_edges_y[0]]

        return stars

    def do_aperture_photometry(self):
        ap = ap_phot(self.img, self.stars)
        ap['xcentroid'] -= 1
        ap['ycentroid'] -= 1
        ap['xcentroid'].name = 'x'
        ap['ycentroid'].name = 'y'
        ap['ap_zpt'] = self.ap_zpt

        return ap

    def do_psf_photometry(self, ap_phot):
        psfphot = psf_phot(self.img, self.psf_model, ap_phot, forced_phot = self.forced_phot)
        psfphot['object_id'] = self.stars['object_id']
        psfphot['pointing'] = self.pointing
        psfphot['sca'] = self.sca
        psfphot['ra'] = self.stars['ra']
        psfphot['dec'] = self.stars['dec']
        psfphot['truth_flux'] = self.stars['flux']
        psfphot['ap_flux'] = ap_phot['aperture_sum']

        exptime = get_exptime(self.band)
        area_eff = roman.collecting_area
        galsim_zp = roman.getBandpasses()[self.band].zeropoint
        truthmag = -2.5*np.log10(psfphot['truth_flux']) + 2.5*np.log10(exptime * area_eff) + galsim_zp
        psfphot['truth_mag'] = truthmag
        psfphot['ap_zpt'] = self.ap_zpt
        psfphot['psf_zpt'] = self.psf_zpt

        return psfphot

    def save_results(self, res, savedir = '.'):
        full_savedir = os.path.join(savedir, f'{self.band}/{self.sca}/')
        os.makedirs(full_savedir, exist_ok=True)
        savepath = os.path.join(full_savedir, f'{self.cutoutsize}_{self.quad_cen_x}_{self.quad_cen_y}_-_{self.zpt_min}_{self.zpt_max}_-_{self.band}_{self.pointing}_{self.sca}.csv')

        print(savepath)

        res.write(savepath, format='csv', overwrite=True)

#################################################################################################################
# RUN BELOW
#################################################################################################################

def run_sca(band, pointing, sca, gridsize, N_stack, zpt_min, zpt_max, forced_phot):
    """
    Just defining a lil' function so I can use multiprocessing.

    """

    grid_eles = np.arange(0, gridsize, 1)
    pairs = list(product(grid_eles, grid_eles))

    imgobj = Image(band, pointing, sca, gridsize, N_stack)
    for pair in pairs:
        i, j = pair
        q = Quad(imgobj, i, j, zpt_min, zpt_max, forced_phot)
        ap = q.do_aperture_photometry()
        psfphot = q.do_psf_photometry(ap)
        q.save_results(psfphot, savedir = '/home/paper_analysis/data/')

        print('Done with pair', pair)

    print('Done with SCA', sca)

    return psfphot

def main():

    """
    Primary called function.

    """

    parser = argparse.ArgumentParser('Do aperture and PSF photometry on a grid')
    parser.add_argument('-b', '--band', type=str, required=True, help='filter')
    parser.add_argument('-p', '--pointing', type=int, required=True, help='Pointing ID')
    parser.add_argument('-s', '--sca', type=int, required=True, help='SCA ID')
    parser.add_argument('-g', '--gridsize', type=int, required=True, help='Size of grid')
    parser.add_argument('-n', '--nstack', type=int, required=False, default=50, help='Number of images to stack per quad')
    parser.add_argument('--min', type=float, required=False, default=19, help='Minimum magnitude used for stars included in generating zeropoint')
    parser.add_argument('--max', type=float, required=False, default=21.5, help='Maximum magnitude used for stars included in generating zeropoint')
    parser.add_argument('--forced', action='store_true', default=False, help='Force photometry at the specified input coordinates.')

    parsedargs = parser.parse_args()

    nprocs = parsedargs.gridsize ** 2

    with Pool(nprocs) as pool:
        pool.apply_async(run_sca, 
                         (parsedargs.band, parsedargs.pointing, parsedargs.sca,
                          parsedargs.gridsize, parsedargs.nstack, 
                          parsedargs.min, parsedargs.max, 
                          parsedargs.forced), 
                         )

        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
    print('All done with everything!')
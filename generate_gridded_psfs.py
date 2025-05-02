# This is intended to be run in Rob's container on Perlmutter.
# See: https://github.com/Roman-Supernova-PIT/phrosty/tree/main/examples/perlmutter

# IMPORTS Standard:
import os
import numpy as np
import pickle as pkl
# from itertools import product
from warnings import warn, simplefilter
# from multiprocessing import Pool
import argparse
import tracemalloc

# IMPORTS Astro:
from astropy.table import Table
from astropy.io import fits
from astropy.nddata import NDData
from astropy.utils.exceptions import AstropyUserWarning
from photutils.background import Background2D
from photutils.psf import extract_stars, EPSFBuilder
from galsim import roman

# IMPORTS Internal:
from phrosty.utils import read_truth_txt, get_exptime

simplefilter('ignore', AstropyUserWarning)

slurmconfig = {'R062': 1,
               'Z087': 2,
               'Y106': 3,
               'J129': 4,
               'H158': 5,
               'F184': 6,
               'K213': 7}

class Grid:
    """
    This carries information about the grid on which the PSF
    is being calculated. It's information that applies to the
    entire chip: grid edges, grid centers, list of images to
    use...

    """

    def __init__(self, band, sca, gridsize, N_stack):

        print('Making grid object.')
        print('Current memory usage:')
        print(tracemalloc.get_traced_memory())

        self.band = band
        self.sca = sca

        self.gridsize = gridsize
        self.cutoutsize = 4088/self.gridsize
        self.grid_edges = np.linspace(self.cutoutsize, 4088, self.gridsize)
        self.grid_centers = np.linspace(0.5 * self.cutoutsize, 4088 - 0.5 * self.cutoutsize, gridsize)

        self.N_stack = N_stack
        self.imagelist = self.get_imagelist()
        self.images, self.psf_pointings = self.get_images(self.imagelist)

        print('Done making grid object.')
        print('Current memory usage:')
        print(tracemalloc.get_traced_memory())
        

    def get_imagelist(self):
        """
        This is a version of the obseq file that only includes pointings you will
        use to generate a PSF. 
        
        These images are used to generate the PSFs for the image you 
        are working on. 

        """

        print('Get image list.')
        print('Current memory usage:')
        print(tracemalloc.get_traced_memory())
        
        # Get a table that doesn't include the image you're working on. 
        obseq_data_path = '/sims_dir/RomanTDS/Roman_TDS_obseq_11_6_23.fits'
        obseq_data = Table.read(obseq_data_path, format='fits')
        obseq_data['pointing'] = np.arange(0, len(obseq_data), 1)
        obseq_data = obseq_data[obseq_data['filter'] == self.band]
        psf_imagelist = obseq_data[:self.N_stack]

        return psf_imagelist

    def get_images(self, imagelist):
        """
        Makes a list of all the images involved in PSF generation 
        and background subtracts them. 

        """
        psf_images = []
        valid_pointings = []
        print('Get images.')
        print('Current memory usage:')
        print(tracemalloc.get_traced_memory())

        for row in imagelist:
            print('Another row in get_images...')
            print('Current memory usage:')
            print(tracemalloc.get_traced_memory())
        
            imgpath = f'/sims_dir/RomanTDS/images/simple_model/{self.band}/{row["pointing"]}/Roman_TDS_simple_model_{self.band}_{row["pointing"]}_{self.sca}.fits.gz'
            try:
                with fits.open(imgpath) as hdu:
                    psfimg_ = hdu[1].data

                bkg = Background2D(psfimg_.data,box_size=(73,73))
                psfimg = psfimg_.data - bkg.background
                psf_images.append(NDData(psfimg))
                valid_pointings.append(row["pointing"])

            except FileNotFoundError:
                print(f'Oops! {self.band} {row["pointing"]} {self.sca} does not exist.')

        print(f'{len(psf_images)}/{self.N_stack} images used for zeropoint stars.')
        return psf_images, valid_pointings

class Quad(Grid):

    """
    Retrieves the grid edges and centers for a given square in the grid,
    where the grid elements are identified by [0:gridsize - 1]. 
    """

    def __init__(self, gridclass, i, j, mag_min, mag_max):

        self.band = gridclass.band
        self.sca = gridclass.sca
        self.cutoutsize = int(gridclass.cutoutsize)
        self.imagelist = gridclass.imagelist
        self.images = gridclass.images
        self.psf_pointings = gridclass.psf_pointings

        self.quad_edges_x = [gridclass.grid_edges[i] - gridclass.cutoutsize, gridclass.grid_edges[i]]
        self.quad_edges_y = [gridclass.grid_edges[j] - gridclass.cutoutsize, gridclass.grid_edges[j]]

        self.quad_cen_x = gridclass.grid_centers[i]
        self.quad_cen_y = gridclass.grid_centers[j]

        self.mag_min = mag_min
        self.mag_max = mag_max

        self.starlists = self.get_starlists()

        print('Made quad object.')
        print('Current memory usage:')
        print(tracemalloc.get_traced_memory())
        

    def get_starlists(self):
        """
        Retrieves a list of stars from the truth tables given the 
        image list from get_imagelist(). These stars are used to generate
        the PSF for a given quad. 
        """

        print('About to get star lists.')
        print('Current memory usage:')
        print(tracemalloc.get_traced_memory())
        
        psftables = []
        for pointing in self.psf_pointings:
            truthtab = read_truth_txt(path = None, band = self.band, pointing = pointing, sca = self.sca)
            truthtab = truthtab[truthtab['obj_type'] == 'star']
            truthtab = truthtab[truthtab['x'] < self.quad_edges_x[1]]
            truthtab = truthtab[truthtab['x'] > self.quad_edges_x[0]]
            truthtab = truthtab[truthtab['y'] < self.quad_edges_y[1]]
            truthtab = truthtab[truthtab['y'] > self.quad_edges_y[0]]

            exptime = get_exptime(self.band)
            area_eff = roman.collecting_area
            galsim_zp = roman.getBandpasses()[self.band].zeropoint
            truthmag = -2.5*np.log10(truthtab['flux']) + 2.5*np.log10(exptime * area_eff) + galsim_zp

            psfmaglim_idx = np.where((truthmag < self.mag_max) & (truthmag > self.mag_min))[0]
            truthtab = truthtab[psfmaglim_idx]

            truthtab['truth_mag'] = truthmag[psfmaglim_idx]

            psftables.append(truthtab)

        print('Done getting star lists.')
        print('Current memory usage:')
        print(tracemalloc.get_traced_memory())

        return psftables

    def make_psf(self, min_stars = 5):

        print('About to make the PSF.')
        print('Current memory usage:')
        print(tracemalloc.get_traced_memory())
        
        stars = extract_stars(data=self.images, catalogs=self.starlists, size=25)
        if len(stars) < min_stars:
            warn(f'There are only {len(stars)} stars for the PSF. This aint going to work.')

        else:
            epsf_builder = EPSFBuilder(oversampling=3, maxiters=50, progress_bar=True)
            psf_model, fitted_stars = epsf_builder(stars)
            print(f'There are {len(fitted_stars)} stars built into this PSF.')

            psfdict = {
                       'psf'         : psf_model, 
                       'truth'       : self.starlists,
                       'fitted_stars': fitted_stars, 
                       'N_stars'     : len(fitted_stars),
                       'band'        : self.band,
                       'sca'         : self.sca,
                       'x_edges'     : self.quad_edges_x,
                       'y_edges'     : self.quad_edges_y,
                       'x_cen'       : self.quad_cen_x,
                       'y_cen'       : self.quad_cen_y
                       }

            print('Done making the PSF.')
            print('Current memory usage:')
            print(tracemalloc.get_traced_memory())
        

            return psfdict

    def save_psf(self, psf, savedir = '.'):
        full_savedir = os.path.join(savedir, f'{self.band}/{self.sca}/')
        os.makedirs(full_savedir, exist_ok=True)
        savepath = os.path.join(full_savedir, f'{self.cutoutsize}_{self.quad_cen_x}_{self.quad_cen_y}_-_{self.mag_min}_{self.mag_max}_-_{self.band}_{self.sca}.psf')
        print('PSF savepath', savepath)

        with open(savepath, 'wb') as f:
            pkl.dump(psf, f, protocol = pkl.HIGHEST_PROTOCOL)

#################################################################################################################
# RUN BELOW
#################################################################################################################

def run_quad(grid, i, j, mag_min, mag_max):
    """
    Just defining a lil' function so I can use multiprocessing.

    """    
    q = Quad(grid, i, j, mag_min, mag_max)
    psf = q.make_psf()
    q.save_psf(psf, savedir = '/home/paper_analysis/gridded_psfs/')

    print('Done with pair', i, j)

def main():

    tracemalloc.start()
    print('Started tracemalloc.')
    print('Current memory usage:')
    print(tracemalloc.get_traced_memory())

    parser = argparse.ArgumentParser('Generate PSFs on a grid')
    parser.add_argument('-b', '--band', type=str, required=True, help='filter')
    parser.add_argument('-s', '--sca', type=int, required=True, help='chip ID')
    parser.add_argument('-g', '--gridsize', type=int, required=True, help='Size of grid')
    parser.add_argument('-i', '--pairi', type=int, required=True, help='Which grid element to make a PSF for, i')
    parser.add_argument('-j', '--pairj', type=int, required=True, help='Which grid element to make a PSF for, j')
    parser.add_argument('-n', '--nstack', type=int, required=False, default=50, help='Number of images to stack per quad')
    parser.add_argument('--min', type=float, required=False, default=19, help='Minimum magnitude used for stars included in generating PSF')
    parser.add_argument('--max', type=float, required=False, default=21.5, help='Maximum magnitude used for stars included in generating PSF')

    parsedargs = parser.parse_args()

    g = Grid(parsedargs.band, parsedargs.sca, 
             gridsize = parsedargs.gridsize, 
             N_stack = parsedargs.nstack
             )

    print('Made grid.')
    print('Current memory usage:')
    print(tracemalloc.get_traced_memory())

    run_quad(g, parsedargs.pairi, parsedargs.pairj, parsedargs.min, parsedargs.max)

    print('Ran quad.')
    print('Current memory usage:')
    print(tracemalloc.get_traced_memory())

    # NOTE: I took multiprocessing out of this because of memory required to do the 8x8 grid.
    # It was better to submit each "quad" (grid element) as a separate slurm job. 
    # But, leaving this here for posterity because I ran grid sizes 1, 2, and 4
    # with apply_async. 
    # 
    # grid_eles = np.arange(0, parsedargs.gridsize, 1)
    # pairs = list(product(grid_eles, grid_eles))
    # nprocs = parsedargs.gridsize ** 2
    # with Pool(nprocs) as pool:
    #     for pair in pairs:
    #         print('Starting pair', pair)
    #         pool.apply_async(run_quad, 
    #                          args = (g, pair, parsedargs.min, parsedargs.max),
    #                          )

    #     pool.close()
    #     pool.join()

if __name__ == "__main__":
    main()
    print('All done with everything!')
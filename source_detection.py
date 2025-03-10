# IMPORTS Standard:
import os 
import numpy as np

# IMPORTS Astro:
from astropy.table import Table, join, hstack
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u

# IMPORTS Internal:
from phrosty.utils import read_truth_txt
from phrosty.imagesubtraction import gz_and_ext

def detect_sources(band,pointing,sca):
    """
    This function runs Source Extractor on an OpenUniverse 
    image specified by its band, pointing, and SCA. 

    """

    imgpath = f'/sims_dir/RomanTDS/images/simple_model/{band}/{pointing}/Roman_TDS_simple_model_{band}_{pointing}_{sca}.fits.gz'
    temp_path = f'/tmp/unzipped_{band}_{pointing}_{sca}.fits'
    gz_and_ext(imgpath, temp_path)
    
    os.system(f'sex {temp_path} -CATALOG_NAME /home/paper_analysis/se_catalogs/scatalog_{band}_{pointing}_{sca}.cat')

def ldac_to_astropy(band,pointing,sca):
    """
    Converts Source Extractor output to an astropy table. 
    
    """
    tab = Table.read(f'/home/paper_analysis/se_catalogs/scatalog_{band}_{pointing}_{sca}.cat', format='ascii.sextractor')
    
    return tab

def crossmatch(pi,ti,seplimit=0.1):
    """ Cross-match the truth files from each image (TI) to the corresponding photometry
    file from that image (PI).

    :param ti: Astropy table from a truth file
    :type ti: Astropy table
    :param pi: Astropy table from a photometry file generated from the image with the same band,
                pointing, and sca as ti. 
    :type pi: Astropy table

    :return: Joined truth catalog and measured photometry catalog, 
            so that the measured objects are correctly crossmatched
            to their corresponding rows in the truth catalog. 
    :rtype: astropy.table 
    """

    if 'ra_truth' not in ti.colnames:
        ti['ra'].name = 'ra_truth'
    
    if 'dec_truth' not in ti.colnames:
        ti['dec'].name = 'dec_truth'

    if 'flux_truth' not in ti.colnames:
        ti['flux'].name = 'flux_truth'

    if 'mag_truth' not in ti.colnames:
        ti['mag'].name = 'mag_truth'

    tc = SkyCoord(ra=ti['ra_truth']*u.degree, dec=ti['dec_truth']*u.degree)
    pc = SkyCoord(ra=pi['ALPHA_SKY'], dec=pi['DELTA_SKY']) # Already has units of degrees!
    ti_idx, pi_idx, angsep, dist3d = search_around_sky(tc,pc,seplimit=seplimit*(u.arcsec))

    ti_reduced = ti[ti_idx]
    pi_reduced = pi[pi_idx]

    ti_pi_reduced = hstack([ti_reduced,pi_reduced], join_type='exact')
    ti_x_pi = join(ti,ti_pi_reduced,join_type='outer')
    
    return ti_x_pi

def main(band, pointing, sca):
    """
    Primary called function. 
    
    """

    detect_sources(band,pointing,sca)

    truth_tab = read_truth_txt(path=None,band=band,pointing=pointing,sca=sca)
    se_tab = ldac_to_astropy(band,pointing,sca)

    xmatch = crossmatch(se_tab,truth_tab)

    xmatch.write(f'/home/paper_analysis/se_catalogs/xmatch_{band}_{pointing}_{sca}.cat', format='csv', overwrite=True)


if __name__ == "__main__":

    scas = np.arange(1, 19, 1)
    bands = {'R062': [35083, 35088, 35857, 36242, 36626, 37391, 37395, 37396, 38155, 38160],
             'Z087': [35138, 35912, 36681, 37066, 37446, 37825, 37830, 38210, 38595, 40115],
             'Y106': [35193, 35198, 35967, 36352, 36736, 37501, 37505, 37506, 38265, 38270], 
             'J129': [35248, 36022, 36791, 37176, 37556, 37935, 37940, 38320, 38705, 40225], 
             'H158': [35303, 35308, 36077, 36462, 36846, 37611, 37615, 37616, 38375, 38380], 
             'F184': [35358, 36132, 36901, 37286, 38045, 38050, 38430, 38815, 40335, 40715],
             'K213': [35413, 35418, 36187, 36956, 37725, 38485, 38490, 38865, 40010, 40390]
             }

    for band in bands.keys():
        for i in range(len(bands[band])):
            for sca in scas:
                main(band,bands[band][i],sca)

    print('All done with everything!')
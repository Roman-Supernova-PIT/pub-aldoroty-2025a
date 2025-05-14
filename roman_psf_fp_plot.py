import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
from phrosty.plotting import roman_sca_plot
from matplotlib import rcParams

SIGMA2FWHM = 2*np.sqrt(2*np.log(2.))


def update_rcParams(key, val):
    if key in rcParams:
        rcParams[key] = val


update_rcParams('font.size', 20)
update_rcParams('font.family', 'serif')
if False:
    update_rcParams('xtick.major.size', 8)
    update_rcParams('xtick.labelsize', 'large')
    update_rcParams('xtick.direction', "in")
    update_rcParams('xtick.minor.visible', True)
    update_rcParams('xtick.top', True)
    update_rcParams('ytick.major.size', 8)
    update_rcParams('ytick.labelsize', 'large')
    update_rcParams('ytick.direction', "in")
    update_rcParams('ytick.minor.visible', True)
    update_rcParams('ytick.right', True)
    update_rcParams('xtick.minor.size', 4)
    update_rcParams('ytick.minor.size', 4)
    update_rcParams('xtick.major.pad', 10)
    update_rcParams('ytick.major.pad', 10)
update_rcParams('legend.numpoints', 1)
update_rcParams('mathtext.fontset', 'cm')
update_rcParams('mathtext.rm', 'serif')
update_rcParams('axes.labelsize', 'x-large')
update_rcParams('lines.markersize', 10)
update_rcParams('lines.markeredgewidth', 1)
update_rcParams('lines.markeredgecolor', 'k')


def flipX(sca):
    if sca % 3 != 0:
        return slice(None, None, 1)
    else:
        return slice(None, None, -1)


def flipY(sca):
    if sca % 3 != 0:
        return slice(None, None, -1)
    else:
        return slice(None, None, 1)


def load_all_data(band, usecols, oversampling, path):
    # data_array = [np.loadtxt(f"roman_psf_bp/{band}_{sca}.txt", delimiter=",")[:,0].reshape(6,6).T[:,::-1] for sca in range(1, 19)]
    data_array = [np.loadtxt(f"{path}/{band}_{sca}_{oversampling}x.txt", delimiter=",", usecols=usecols).reshape(6,6).T[::-1,:][flipX(sca),flipY(sca)] for sca in range(1, 19)]
    data_array = np.array(data_array)
    if usecols == 2:
        data_array *= SIGMA2FWHM*0.11
    return data_array


def determine_vlims(data_array, usecols):
    if usecols in {0, 1, 10+2, 14+2}:
        vmin, vmax = data_array.min(), data_array.max()
    elif usecols in {2,}:
        vmin, vmax = 0.119, 0.143
    elif usecols in {3, 4}:
        vmin = -0.23
        vmax = -vmin
    else:
        vmax = np.abs(data_array).max()
        vmin = -vmax

    return vmin, vmax


def main():
    parser = argparse.ArgumentParser("Roman PSF plotting on the Focal Plane")
    parser.add_argument("--source", type=str, required=True, help="Path to the folder containing the source data")
    parser.add_argument("--destination", type=str, required=False, default="./", help="Path to the folder to save the plots")
    parser.add_argument("--band", type=str, required=True, help="Name of the bandpass")
    parser.add_argument("--usecols", type=int, required=True,
                        help="The column number to plot. See roman_psf_fp.py. "
                             "clabel might have to be adjusted accordingly."
                        )
    parser.add_argument("--clabel", type=str, required=False,
                        default=r"PSF FWHM [arcsec]", help="Label on the color bar")
    parser.add_argument("--oversampling", type=float, required=False, default=8.0, help="Oversampling factor")
    args = parser.parse_args()

    path = args.source
    destination = args.destination
    band = args.band
    usecols = int(args.usecols)
    clabel = args.clabel
    oversampling = float(args.oversampling)

    data_array = load_all_data(band=band, usecols=usecols, oversampling=oversampling, path=path)
    vlims = determine_vlims(data_array, usecols)

    roman_sca_plot(
        data_array,
        np.array(list(range(1, 19))),
        ptype="image",
        residual_plot=False,
        savefig=True,
        show_sca_id=True,
        title=band,
        vlims=vlims,
        clabel=clabel,
        cmap=cm.cividis,
        savepath=f"{destination}/{band}.pdf")


if __name__ == "__main__":
    main()

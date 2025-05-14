import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
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


def roman_sca_plot(data_array, sca_order, ptype='image', cmap='bwr', residual_plot=True, clabel=None, title=None, vlims=None,
                   return_fig=False, savefig=False, show_sca_id=False, savepath='roman_scas.png'):
    from astropy.visualization import ZScaleInterval
    detector = plt.figure(figsize=(10,6),dpi=300)
    nrows, ncols = 55,91
    grid = detector.add_gridspec(nrows=nrows,ncols=ncols,figure=detector,
                                 width_ratios=[1]*ncols, height_ratios=[1]*nrows,
                                 hspace=0,wspace=0.1)
    row_begins = np.array([10,3,0,0,3,10])
    row_ends = np.array([x+14 for x in row_begins])
    col_begins = np.arange(0,ncols,14)
    col_ends = np.array([x+14 for x in col_begins])
    add_distance = [15,16,16]

    axs = []
    for row in add_distance:
        for i in range(len(row_begins)):
            ax = detector.add_subplot(grid[row_begins[i]:row_ends[i],col_begins[i]+1:col_ends[i]])
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
            axs.append(ax)

        row_begins += row
        row_ends += row

    # Argument data_array should be an array of len(N SCAs) containing arrays:
    # fake_data = np.array([np.random.rand(14,14)]*len(axs))
    if ptype=='image':
        if vlims is None:
            vmin, vmax = ZScaleInterval().get_limits(data_array.ravel())
        else:
            vmin, vmax = vlims

    sortidx = sca_order.argsort()
    sca_order = sca_order[sortidx]
    data_array = data_array[sortidx]
    imsim_sca_order = np.array([9,6,3,12,15,18,8,5,2,11,14,17,7,4,1,10,13,16])-1

    for i, sca in enumerate(imsim_sca_order):
        if ptype=='image':
            if residual_plot:
                ends = np.nanmax(np.array([abs(vmin),abs(vmax)]))
                im = axs[i].imshow(data_array[sca], cmap=cmap,
                                   norm=MidpointNormalize(midpoint=0,vmin=-ends,vmax=ends))
            else:
                im = axs[i].imshow(data_array[sca], cmap=cmap, vmin=vmin,vmax=vmax, interpolation="spline16")
        elif ptype=='scatter':
            axs[i].scatter(data_array[sca][:,0],data_array[sca][:,1], c=data_array[sca][:,2], marker='.',linestyle='')
            if residual_plot:
                axs[i].axhline(0,color='k',linestyle='--')

        if show_sca_id:
            axs[i].annotate(sca+1, xy=(0,1), fontsize=12)

    if ptype=='image':
        cbar_ax = detector.add_subplot(grid[:,-4:-1])
        cbar = plt.colorbar(im, cax=cbar_ax)
        if clabel is not None:
            cbar.set_label(clabel, labelpad=20, fontsize=18, rotation=270)
    if title is not None:
        plt.suptitle(title, y=0.93, fontsize=18)
    if savefig:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    if return_fig:
        return detector
    else:
        plt.show()


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
    parser.add_argument("--path", type=str, required=True, help="Path to the folder containing the data")
    parser.add_argument("--band", type=str, required=True, help="Name of the bandpass")
    parser.add_argument("--usecols", type=int, required=True,
                        help="The column number to plot. See roman_psf_fp.py. "
                             "clabel might have to be adjusted accordingly."
                        )
    parser.add_argument("--clabel", type=str, required=False,
                        default=r"PSF FWHM [arcsec]", help="Label on the color bar")
    parser.add_argument("--oversampling", type=float, required=False, default=8.0, help="Oversampling factor")
    args = parser.parse_args()

    path = args.path
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
        savepath=f"{band}.pdf")


if __name__ == "__main__":
    main()

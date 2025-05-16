import argparse
import galsim
import galsim.roman
import numpy as np
import logging
from itertools import product

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser("Calculate PSF properties on the Roman Focal Plane")
parser.add_argument("--band", type=str, required=True, help="Name of the band")
parser.add_argument("--SCA", type=int, required=True, choices=list(range(1, 18)), help="SCA ID")
parser.add_argument("--oversampling", type=float, default=8.0, required=False, help="Oversampling factor")
parser.add_argument("--step", type=int, default=800, required=False,
                    help="How many pixels to stride before evaluating PSFs again"
                    )
args = parser.parse_args()
bp_name = args.band
SCA = int(args.SCA)
step = int(args.step)

oversampling_factor = float(args.oversampling)
n_waves = None

bp_dict = galsim.roman.getBandpasses()
bp = bp_dict[bp_name]

chromatic_psf = galsim.roman.getPSF(SCA=SCA, bandpass=bp_name, n_waves=n_waves, SCA_pos=galsim._PositionD(4088, 4088))

size_list, e1_list, e2_list = [], [], []
x_list, y_list = [], []
aber_list = []
for x, y in product(range(0, galsim.roman.n_pix, step), range(0, galsim.roman.n_pix, step)):
    i, j = x//step, y//step
    logging.info("Calculating PSF for i=%d, j=%d for SCA=%d", i, j, SCA)
    chromatic_psf = galsim.roman.getPSF(SCA=SCA, bandpass=bp_name, n_waves=n_waves, SCA_pos=galsim._PositionD(x, y))
    achromatic_psf = chromatic_psf.evaluateAtWavelength(bp.effective_wavelength)
    aberrations = achromatic_psf._screen.aberrations
    epsf = galsim.Convolve([achromatic_psf, galsim.Pixel(galsim.roman.pixel_scale)])
    im = epsf.drawImage(bandpass=bp, scale=galsim.roman.pixel_scale/oversampling_factor, method="no_pixel")
    shape = galsim.hsm.FindAdaptiveMom(im, strict=False)
    x_list.append(x)
    y_list.append(y)
    size_list.append(shape.moments_sigma/oversampling_factor)
    e1_list.append(shape.observed_e1)
    e2_list.append(shape.observed_e2)
    aber_list.append(aberrations)

output_file = f"roman_psf_bp/{bp_name}_{SCA}_{oversampling_factor}x.txt"
np.savetxt(output_file, np.hstack([np.array([x_list, y_list, size_list, e1_list, e2_list]).T, np.array(aber_list)]), fmt="%d, %d, %.6f, %.6f, %.6f, " + "%.8f, "*23)


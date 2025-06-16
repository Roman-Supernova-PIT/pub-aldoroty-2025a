"Initial Characterization of Photometry of High-Latitude Time Domain Survey _Roman_ images from the OpenUniverse Simulations"
Aldoroty, L.; Scolnic, D.; Kannawadi, A.; Troxel, M. and the Roman Supernova Project Infrastructure Team 2025.

* data/[band]/511_[band]_all.parquet: Photometric results for each band for the 8x8 grids.
* generate_gridded_psfs.py: Generates the PSFs used in the above work. 
* gridded_stellar_phot.py: Uses the PSFs from generate_gridded_psfs.py to measure stellar fluxes.
* psf_sca_plots.py: Generates the plot in Figure 10. Calculates the asymmetry indices in Table 4. Retrieves e1, e2 for all PSFs at the center of each SCA.
* roman_psf_fp.py: Retrieve information for Figures 1 and 9. 
* roman_psf_fp_plot.py: Plot Figures 1 and 9. 
* source_detection.py: Uses Source Extractor on default settings to detect stars.

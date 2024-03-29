import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys, time
from functions import load_data, prepare_spectrum, mcmc, point_estimates, plotting
from spd_setup import spd_setup
t0 = time.time()

target_num = int(sys.argv[1])   # used to select array of spectra to analyse

var = spd_setup()   # instantiate params

# get mastar data and targets to analyse
mast_data = load_data(number=target_num)
targets = mast_data.get_targets()
mast_data.get_mastar()

# get estimates data
mast_data.get_estimates()
ebv_gaia = mast_data.meta_data['ebv']

print(targets)
for c, i in enumerate(targets):
    print('Running spectrum: {}'.format(mast_data.pim[c]))

    # interpolate (if necessary), de-redden and median normalise spectrum
    clean_spec = prepare_spectrum(wave=mast_data.wave, flux=mast_data.flux[c], ivar=mast_data.ivar[c],
                                  ebv=ebv_gaia[c], spec_id=c)
    clean_spec.get_med_data()

    if clean_spec.catch_remaining(meta_data=mast_data.meta_data[c]):
        # save data to blank file
        continue

    # run mcmc for BOSZ models
    mcmc_run = mcmc(flux=clean_spec.corrected_flux_med, yerr=clean_spec.yerr, meta_data=mast_data.meta_data[c])
    mcmc_run.starting()     # starting values for walkers
    mcmc_run.sample(model='BOSZ')     # use emcee to sample param space

    # get point estimates from chains
    point_bosz = point_estimates(clean_spec, mast_data.pim[c], model='BOSZ')
    point_bosz.flatchain(mcmc_run)
    point_bosz.params_err()
    point_bosz.get_model_fit()
    point_bosz.get_chi2()
    point_bosz.save_data()

    plot_bosz = plotting(point_bosz, clean_spec, mast_data.pim[c], c, model='BOSZ')

    if mast_data.meta_data[c]['minTEFF_gaia'] > 5000:       # only use marcs for low Teff
        continue    # continue to next spectrum in for loop

    # run mcmc for MARCS models
    mcmc_run.sample(model='MARCS')  # use emcee to sample param space

    # get point estimates from chains
    point_marcs = point_estimates(clean_spec, mast_data.pim[c], model='MARCS')
    point_marcs.flatchain(mcmc_run)
    point_marcs.params_err()
    point_marcs.get_model_fit()
    point_marcs.get_chi2()
    point_marcs.save_data()

    plot_marcs = plotting(point_marcs, clean_spec, mast_data.pim[c], c, model='MARCS')

print('\nTotal time taken: ', time.time() - t0)

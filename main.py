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
t0 = time.time()

# instantiate params
var = spd_setup()

# get mastar data and targets to analyse
mast_data = load_data()
targets = mast_data.get_targets(number=target_num)
mast_data.get_mastar(start=targets[0], end=targets[-1])

# get estimates data
mast_data.get_estimates(start=targets[0], end=targets[-1])
ebv_gaia = mast_data.meta_data['ebv']

print('\nT1: ', time.time()-t0)

for c, i in enumerate(targets):
    print('Running spectrum: {}'.format(i), end='\r')

    # interpolate (if necessary), de-redden and median normalise spectrum
    clean_spec = prepare_spectrum(wave=mast_data.wave, flux=mast_data.flux[c][9:-8], ivar=mast_data.ivar[c],
                                  ebv=ebv_gaia[c], spec_id=c)
    clean_spec.get_med_data()
    print('\nT2: ', time.time() - t0)

    if clean_spec.catch_remaining(meta_data=mast_data.meta_data[c]):
        # save data to blank file
        continue

    # run mcmc for BOSZ models
    mcmc_run = mcmc(flux=clean_spec.corrected_flux_med, yerr=clean_spec.yerr, meta_data=mast_data.meta_data[c],
                    parallel=True)
    mcmc_run.starting()     # starting values for walkers
    mcmc_run.sample(model='BOSZ')     # use emcee to sample param space

    # get point estimates from chains
    point = point_estimates(clean_spec, mast_data.pim[c])
    point.flatchain(mcmc_run, model='BOSZ')
    point.params_err(model='BOSZ')
    point.get_model_fit(model='BOSZ')
    point.get_chi2(model='BOSZ')

    point.save_data(model='BOSZ')
    plot_bosz = plotting(point, clean_spec, mast_data, c, model='BOSZ')

    if mast_data.meta_data[c]['minTEFF_gaia'] > 5000:       # only use marcs for low Teff
        continue

    # run mcmc for MARCS models
    mcmc_run.sample(model='MARCS')  # use emcee to sample param space

    # get point estimates from chains
    point.flatchain(mcmc_run, model='MARCS')
    point.params_err(model='MARCS')
    point.get_model_fit(model='MARCS')
    point.get_chi2(model='MARCS')

    point.save_data(model='MARCS')
    plot_marcs = plotting(point, clean_spec, mast_data, c, model='MARCS')

print('\nTotal time taken: ', time.time() - t0)

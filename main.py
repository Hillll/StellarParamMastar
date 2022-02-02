'''
  mcmc stellar parameter determination with pPXF ST to evaluate likelihood
- tophat priors from gaia photometry
- incorporation of errors in ppxf and used in calculating reduced chi**2
- zero values in ivar are interpolated over to fill gaps
- use of pPXF V7 
- v11-1 correct error
- correct marcs models
Fit mastar data with bosz and marcs models independtly and record corresponding chi2

'''
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

for c, i in enumerate(targets[:1]):
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
    temp = mcmc(model='BOSZ', flux=clean_spec.corrected_flux_med, yerr=clean_spec.yerr,
                     meta_data=mast_data.meta_data[c], parallel=True)
    temp.starting()     # starting values for walkers
    sampler = temp.sample()     # use emcee to sample param space

    # get point estimates from chains
    point = point_estimates(sampler, clean_spec, mast_data.pim[c])
    point.params_err(model='BOSZ')
    point.get_model_fit(model='BOSZ')
    point.get_chi2(model='BOSZ')

    if mast_data.meta_data[c]['minTEFF_gaia'] > 5000:       #skip to next object if min teff est is high
        point.save_data(model='BOSZ')
        if var.plot:
            plot_bosz = plotting(sampler, point, clean_spec, mast_data, c, model='BOSZ')

    # run mcmc for MARCS models
    temp = mcmc(model='MARCS', flux=clean_spec.corrected_flux_med, yerr=clean_spec.yerr,
                meta_data=mast_data.meta_data[c], parallel=True)
    temp.starting()  # starting values for walkers
    sampler = temp.sample()  # use emcee to sample param space

    # get point estimates from chains
    point = point_estimates(sampler, clean_spec, mast_data.pim[c], model='MARCS')
    point.params_err(teff='median', logg='median', zh='mode', alpha='mode')
    point.get_model_fit()
    point.get_chi2()
    point.save_data()

    plot_marcs = plotting(sampler, point, clean_spec, mast_data, c, model='MARCS')

print('\nTotal time taken: ', time.time() - t0)
'''

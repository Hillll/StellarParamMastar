import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys, time
from functions import load_data, prepare_spectrum_solar, mcmc_solar, point_estimates, plotting
from spd_setup import spd_setup

t0 = time.time()

target_num = int(sys.argv[1])   # used to select array of spectra to analyse
t0 = time.time()

# instantiate params
var = spd_setup()

# get mastar data and targets to analyse
sol_data = load_data()
sol_data.get_solar()

# get estimates data
ebv_gaia = 0

print('\nT1: ', time.time()-t0)
print('Running spectrum')

# interpolate (if necessary), de-redden and median normalise spectrum
clean_spec = prepare_spectrum_solar(wave=sol_data.wave[9:-8], flux=sol_data.flux[9:-8], yerr=sol_data.yerr[9:-8])
clean_spec.get_med_data()
print('\nT2: ', time.time() - t0)

# run mcmc for BOSZ models
mcmc_run = mcmc_solar(flux=clean_spec.corrected_flux_med, yerr=clean_spec.yerr, parallel=True)
mcmc_run.starting()     # starting values for walkers
mcmc_run.sample(model='BOSZ')     # use emcee to sample param space

# get point estimates from chains
point = point_estimates(clean_spec, 999)
point.flatchain(mcmc_run, model='BOSZ')
point.params_err(model='BOSZ')
point.get_model_fit(model='BOSZ')
point.get_chi2(model='BOSZ')

point.save_data(model='BOSZ')
plot_bosz = plotting(point, clean_spec, 999, 0, model='BOSZ')

# run mcmc for MARCS models
mcmc_run.sample(model='MARCS')  # use emcee to sample param space

# get point estimates from chains
point.flatchain(mcmc_run, model='MARCS')
point.params_err(model='MARCS')
point.get_model_fit(model='MARCS')
point.get_chi2(model='MARCS')

point.save_data(model='MARCS')
plot_marcs = plotting(point, clean_spec, 999, 0, model='MARCS')

print('\nTotal time taken: ', time.time() - t0)

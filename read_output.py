"""Read output files from MCMC run."""

import sys, os
import numpy as np
from astropy.io import fits
from spd_setup import spd_setup
import glob, pickle

var = spd_setup()

# get IDs from input file
header = fits.open(var.data_direc + var.spec_file)
mangaid = header[1].data['mangaid']
plate, ifu, mjd = header[1].data['plate'], header[1].data['ifudesign'], header[1].data['mjd']
ifu = np.asarray([int(i) for i in ifu])   # ensure ifu is int
pim_input = [int(str((plate[i])) + str((ifu[i])) + str((mjd[i]))) for i in range(len(plate))]
header.close()

# create arrays to be populated
teff_marcs, teff_err_dn_marcs, teff_err_up_marcs = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0), \
                                                   np.full(len(pim_input), -999.0)
teff_bosz, teff_err_dn_bosz, teff_err_up_bosz = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0), \
                                                   np.full(len(pim_input), -999.0)

logg_marcs, logg_err_dn_marcs, logg_err_up_marcs = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0), \
                                                   np.full(len(pim_input), -999.0)
logg_bosz, logg_err_dn_bosz, logg_err_up_bosz = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0), \
                                                   np.full(len(pim_input), -999.0)

met_marcs, met_err_dn_marcs, met_err_up_marcs = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0), \
                                                   np.full(len(pim_input), -999.0)
met_bosz, met_err_dn_bosz, met_err_up_bosz = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0), \
                                                   np.full(len(pim_input), -999.0)

alp_marcs, alp_err_dn_marcs, alp_err_up_marcs = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0), \
                                                   np.full(len(pim_input), -999.0)
alp_bosz, alp_err_dn_bosz, alp_err_up_bosz = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0), \
                                                   np.full(len(pim_input), -999.0)

chi_marcs, chi_bosz = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0)
converged_marcs, converged_bosz = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0)
iterations_marcs, iterations_bosz = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0)

teff_best, teff_err_dn_best, teff_err_up_best = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0), \
                                                   np.full(len(pim_input), -999.0)
logg_best, logg_err_dn_best, logg_err_up_best = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0), \
                                                   np.full(len(pim_input), -999.0)
met_best, met_err_dn_best, met_err_up_best = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0), \
                                                   np.full(len(pim_input), -999.0)
alp_best, alp_err_dn_best, alp_err_up_best = np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0), \
                                                   np.full(len(pim_input), -999.0)
chi_best, model_best, valid, converged_best = np.full(len(pim_input), -999.0), np.full(len(pim_input), 'N'), \
                              np.full(len(pim_input), -999.0), np.full(len(pim_input), -999.0)

# loop through pim_input and populate arrays with data
for c, i in enumerate(pim_input[:1000]):
    print(c, end='\r')
    # load output data
    output = glob.glob(var.output_direc + str(i) + '*')
    if len(output) == 0:
        continue
    elif len(output) == 1:      # must be a bosz file but add marcs option
        if 'MARCS' in output[0]:
            model = 'MARCS'
            with open(var.output_direc + str(i) + '_' + model + '.pkl', 'rb') as f:
                data = pickle.load(f)
            params = data['MARCS_params']
            teff_marcs[c], teff_err_dn_marcs[c], teff_err_up_marcs[c] = params[0][0], params[0][1], params[0][2]
            logg_marcs[c], logg_err_dn_marcs[c], logg_err_up_marcs[c] = params[1][0], params[1][1], params[1][2]
            met_marcs[c], met_err_dn_marcs[c], met_err_up_marcs[c] = params[2][0], params[2][1], params[2][2]
            alp_marcs[c], alp_err_dn_marcs[c], alp_err_up_marcs[c] = params[3][0], params[3][1], params[3][2]
            chi_marcs[c], converged_marcs[c] = data['MARCS_chi'], data['MARCS_converged']
            iterations_marcs[c] = data['MARCS_samples'].shape[1] + var.burnin

            teff_best[c], teff_err_dn_best[c], teff_err_up_best[c] = params[0][0], params[0][1], params[0][2]
            logg_best[c], logg_err_dn_best[c], logg_err_up_best[c] = params[1][0], params[1][1], params[1][2]
            met_best[c], met_err_dn_best[c], met_err_up_best[c] = params[2][0], params[2][1], params[2][2]
            alp_best[c], alp_err_dn_best[c], alp_err_up_best[c] = params[3][0], params[3][1], params[3][2]
            model_best[c] = 'M'
            chi_best[c], converged_best[c] = data['MARCS_chi'], data['MARCS_converged']

        elif 'BOSZ' in output[0]:
            model = 'BOSZ'
            with open(var.output_direc + str(i) + '_' + model + '.pkl', 'rb') as f:
                data = pickle.load(f)
            params = data['BOSZ_params']
            teff_bosz[c], teff_err_dn_bosz[c], teff_err_up_bosz[c] = params[0][0], params[0][1], params[0][2]
            logg_bosz[c], logg_err_dn_bosz[c], logg_err_up_bosz[c] = params[1][0], params[1][1], params[1][2]
            met_bosz[c], met_err_dn_bosz[c], met_err_up_bosz[c] = params[2][0], params[2][1], params[2][2]
            alp_bosz[c], alp_err_dn_bosz[c], alp_err_up_bosz[c] = params[3][0], params[3][1], params[3][2]
            chi_bosz[c], converged_bosz[c] = data['BOSZ_chi'], data['BOSZ_converged']
            iterations_bosz[c] = data['BOSZ_samples'].shape[1] + var.burnin

            teff_best[c], teff_err_dn_best[c], teff_err_up_best[c] = params[0][0], params[0][1], params[0][2]
            logg_best[c], logg_err_dn_best[c], logg_err_up_best[c] = params[1][0], params[1][1], params[1][2]
            met_best[c], met_err_dn_best[c], met_err_up_best[c] = params[2][0], params[2][1], params[2][2]
            alp_best[c], alp_err_dn_best[c], alp_err_up_best[c] = params[3][0], params[3][1], params[3][2]
            model_best[c] = 'B'
            chi_best[c], converged_best[c] = data['BOSZ_chi'], data['BOSZ_converged']

    elif len(output) == 2:
        model = 'MARCS'
        with open(var.output_direc + str(i) + '_' + model + '.pkl', 'rb') as f:
            data_marcs = pickle.load(f)
        params_marcs = data_marcs['MARCS_params']
        teff_marcs[c], teff_err_dn_marcs[c], teff_err_up_marcs[c] = params_marcs[0][0], params_marcs[0][1], params_marcs[0][2]
        logg_marcs[c], logg_err_dn_marcs[c], logg_err_up_marcs[c] = params_marcs[1][0], params_marcs[1][1], params_marcs[1][2]
        met_marcs[c], met_err_dn_marcs[c], met_err_up_marcs[c] = params_marcs[2][0], params_marcs[2][1], params_marcs[2][2]
        alp_marcs[c], alp_err_dn_marcs[c], alp_err_up_marcs[c] = params_marcs[3][0], params_marcs[3][1], params_marcs[3][2]
        chi_marcs[c], converged_marcs[c] = data_marcs['MARCS_chi'], data_marcs['MARCS_converged']
        iterations_marcs[c] = data_marcs['MARCS_samples'].shape[1] + var.burnin

        model = 'BOSZ'
        with open(var.output_direc + str(i) + '_' + model + '.pkl', 'rb') as f:
            data_bosz = pickle.load(f)
        params_bosz = data_bosz['BOSZ_params']
        teff_bosz[c], teff_err_dn_bosz[c], teff_err_up_bosz[c] = params_bosz[0][0], params_bosz[0][1], params_bosz[0][2]
        logg_bosz[c], logg_err_dn_bosz[c], logg_err_up_bosz[c] = params_bosz[1][0], params_bosz[1][1], params_bosz[1][2]
        met_bosz[c], met_err_dn_bosz[c], met_err_up_bosz[c] = params_bosz[2][0], params_bosz[2][1], params_bosz[2][2]
        alp_bosz[c], alp_err_dn_bosz[c], alp_err_up_bosz[c] = params_bosz[3][0], params_bosz[3][1], params_bosz[3][2]
        chi_bosz[c], converged_bosz[c] = data_bosz['BOSZ_chi'], data_bosz['BOSZ_converged']
        iterations_bosz[c] = data_bosz['BOSZ_samples'].shape[1] + var.burnin

        # best model based on chi2
        if data_marcs['MARCS_chi'] < data_bosz['BOSZ_chi']:
            teff_best[c], teff_err_dn_best[c], teff_err_up_best[c] = params_marcs[0][0], params_marcs[0][1], params_marcs[0][2]
            logg_best[c], logg_err_dn_best[c], logg_err_up_best[c] = params_marcs[1][0], params_marcs[1][1], params_marcs[1][2]
            met_best[c], met_err_dn_best[c], met_err_up_best[c] = params_marcs[2][0], params_marcs[2][1], params_marcs[2][2]
            alp_best[c], alp_err_dn_best[c], alp_err_up_best[c] = params_marcs[3][0], params_marcs[3][1], params_marcs[3][2]
            model_best[c] = 'M'
            chi_best[c], converged_best[c] = data_marcs['MARCS_chi'], data_marcs['MARCS_converged']
        elif data_marcs['MARCS_chi'] > data_bosz['BOSZ_chi']:
            teff_best[c], teff_err_dn_best[c], teff_err_up_best[c] = params_bosz[0][0], params_bosz[0][1], params_bosz[0][2]
            logg_best[c], logg_err_dn_best[c], logg_err_up_best[c] = params_bosz[1][0], params_bosz[1][1], params_bosz[1][2]
            met_best[c], met_err_dn_best[c], met_err_up_best[c] = params_bosz[2][0], params_bosz[2][1], params_bosz[2][2]
            alp_best[c], alp_err_dn_best[c], alp_err_up_best[c] = params_bosz[3][0], params_bosz[3][1], params_bosz[3][2]
            model_best[c] = 'B'
            chi_best[c], converged_best[c] = data_bosz['BOSZ_chi'], data_bosz['BOSZ_converged']


    # create valid column
    if chi_best[c] == -999:
        continue
    elif chi_best[c] < 30:
        valid[c] = 1
    elif chi_best[c] > 30 and teff_best[c] < 4000:
        valid[c] = 1
    else:
        valid[c] = 0


c1 = fits.Column(name='MANGAID', array=mangaid, format='60A')
c2 = fits.Column(name='PLATE', array=plate, format='K')
c3 = fits.Column(name='IFU', array=ifu, format='K')
c4 = fits.Column(name='MJD', array=mjd, format='K')
c5 = fits.Column(name='PIM', array=pim_input, format='K')

c6 = fits.Column(name='teff_marcs', array=teff_marcs, format='D')
c7 = fits.Column(name='teff_err_dn_marcs', array=teff_err_dn_marcs, format='D')
c8 = fits.Column(name='teff_err_up_marcs', array=teff_err_up_marcs, format='D')
c9 = fits.Column(name='teff_bosz', array=teff_bosz, format='D')
c10 = fits.Column(name='teff_err_dn_bosz', array=teff_err_dn_bosz, format='D')
c11 = fits.Column(name='teff_err_up_bosz', array=teff_err_up_bosz, format='D')

c12 = fits.Column(name='logg_marcs', array=logg_marcs, format='D')
c13 = fits.Column(name='logg_err_dn_marcs', array=logg_err_dn_marcs, format='D')
c14 = fits.Column(name='logg_err_up_marcs', array=logg_err_up_marcs, format='D')
c15 = fits.Column(name='logg_bosz', array=logg_bosz, format='D')
c16 = fits.Column(name='logg_err_dn_bosz', array=logg_err_dn_bosz, format='D')
c17 = fits.Column(name='logg_err_up_bosz', array=logg_err_up_bosz, format='D')

c18 = fits.Column(name='mh_marcs', array=met_marcs, format='D')
c19 = fits.Column(name='mh_err_dn_marcs', array=met_err_dn_marcs, format='D')
c20 = fits.Column(name='mh_err_up_marcs', array=met_err_up_marcs, format='D')
c21 = fits.Column(name='mh_bosz', array=met_bosz, format='D')
c22 = fits.Column(name='mh_err_dn_bosz', array=met_err_dn_bosz, format='D')
c23 = fits.Column(name='mh_err_up_bosz', array=met_err_up_bosz, format='D')

c24 = fits.Column(name='am_marcs', array=alp_marcs, format='D')
c25 = fits.Column(name='am_err_dn_marcs', array=alp_err_dn_marcs, format='D')
c26 = fits.Column(name='am_err_up_marcs', array=alp_err_up_marcs, format='D')
c27 = fits.Column(name='am_bosz', array=alp_bosz, format='D')
c28 = fits.Column(name='am_err_dn_bosz', array=alp_err_dn_bosz, format='D')
c29 = fits.Column(name='am_err_up_bosz', array=alp_err_up_bosz, format='D')

c30 = fits.Column(name='chi_bosz', array=chi_bosz, format='D')
c31 = fits.Column(name='chi_marcs', array=chi_marcs, format='D')

c32 = fits.Column(name='iterations_marcs', array=iterations_marcs, format='D')
c33 = fits.Column(name='iterations_bosz', array=iterations_bosz, format='D')

c34 = fits.Column(name='teff_best', array=teff_best, format='D')
c35 = fits.Column(name='teff_err_dn_best', array=teff_err_dn_best, format='D')
c36 = fits.Column(name='teff_err_up_best', array=teff_err_up_best, format='D')
c37 = fits.Column(name='logg_best', array=logg_best, format='D')
c38 = fits.Column(name='logg_err_dn_best', array=logg_err_dn_best, format='D')
c39 = fits.Column(name='logg_err_up_best', array=logg_err_up_best, format='D')
c40 = fits.Column(name='mh_best', array=met_best, format='D')
c41 = fits.Column(name='mh_err_dn_best', array=met_err_dn_best, format='D')
c42 = fits.Column(name='mh_err_up_best', array=met_err_up_best, format='D')
c43 = fits.Column(name='am_best', array=alp_best, format='D')
c44 = fits.Column(name='am_err_dn_best', array=alp_err_dn_best, format='D')
c45 = fits.Column(name='am_err_up_best', array=alp_err_up_best, format='D')

c46 = fits.Column(name='chi_best', array=chi_best, format='D')
c47 = fits.Column(name='converged_best', array=converged_best, format='D')
c48 = fits.Column(name='model_best', array=model_best, format='10A')
c49 = fits.Column(name='valid', array=valid, format='D')

t = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
                                   c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36,
                                   c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49])

t.writeto(var.output_direc + 'mpl11_v-1_7_7-output_params.fits', overwrite=True)


















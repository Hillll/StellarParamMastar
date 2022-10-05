"""Retrospectively plot the fits, corner and trace of an MCMC run"""

from plotting_retro_class import load_data, prepare_spectrum, plotting, point_estimates
import glob, pickle
import numpy as np

data_direc = '/home/lewishill/PycharmProjects/SPD/output/'
output_folder = '/home/lewishill/PycharmProjects/SPD/plots/'

ids = np.array([1000170157372, 1000170157373, 1000170257372])

mast_data = load_data(pims_todo=ids)
mast_data.get_mastar()
mast_data.get_estimates()
ebv_gaia = mast_data.meta_data['ebv']


for c, i in enumerate(ids):
    print(c, end='\r')
    # clean spec
    clean_spec = prepare_spectrum(wave=mast_data.wave, flux=mast_data.flux[c][9:-8], ivar=mast_data.ivar[c],
                                  ebv=ebv_gaia[c], spec_id=c)
    clean_spec.get_med_data()
    # load output data
    output = glob.glob(data_direc + str(i) + '*.pkl')
    if len(output) == 0:
        print('\nNo data available.')

    
    elif len(output) == 1:
        if 'MARCS' in output[0]:
            model = 'MARCS'
            with open(data_direc + str(i) + '_' + model + '.pkl', 'rb') as f:
                data = pickle.load(f)
            params = data['MARCS_params']
            samples = data['MARCS_samples']
            point = point_estimates(clean_spec, samples, i, 'MARCS')
            point.params_err()
            point.get_model_fit()
            point.get_chi2()
            plotting(point, clean_spec, samples, pim=i, model='MARCS', output_folder=output_folder)
        elif 'BOSZ' in output[0]:
            model = 'BOSZ'
            with open(data_direc + str(i) + '_' + model + '.pkl', 'rb') as f:
                data = pickle.load(f)
            params = data['BOSZ_params']
            samples = data['BOSZ_samples']
            point = point_estimates(clean_spec, samples, i, 'BOSZ')
            point.params_err()
            point.get_model_fit()
            point.get_chi2()
            plotting(point, clean_spec, samples, pim=i, model='BOSZ', output_folder=output_folder)

    elif len(output) == 2:
        model = 'MARCS'
        with open(data_direc + str(i) + '_' + model + '.pkl', 'rb') as f:
            data = pickle.load(f)
        params = data['MARCS_params']
        samples = data['MARCS_samples']
        point = point_estimates(clean_spec, samples, i, 'MARCS')
        point.params_err()
        point.get_model_fit()
        point.get_chi2()
        plotting(point, clean_spec, samples, pim=i, model='MARCS', output_folder=output_folder)

        model = 'BOSZ'
        with open(data_direc + str(i) + '_' + model + '.pkl', 'rb') as f:
            data = pickle.load(f)
        params = data['BOSZ_params']
        samples = data['BOSZ_samples']
        point = point_estimates(clean_spec, samples, i, 'BOSZ')
        point.params_err()
        point.get_model_fit()
        point.get_chi2()
        plotting(point, clean_spec, samples, pim=i, model='BOSZ', output_folder=output_folder)
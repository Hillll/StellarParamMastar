'''Generate a model spectrum of MARCS of BOSZ'''

import numpy as np
from scipy.interpolate import interp1d
from interp import interp_models

def model_spec(theta, model):  # model given a set of parameters (theta)
    if alpha == True:
        t, g, z, a = theta
        ap = (t, g, z)
        if model == 'bosz' or model == 'BOSZ':
            flux_m03 = bosz_m03.generate_stellar_spectrum(ap)  # get alpha model for each combination of t,g,z
            flux_p0 = bosz_p0.generate_stellar_spectrum(ap)
            flux_p03 = bosz_p03.generate_stellar_spectrum(ap)
            flux_p05 = bosz_p05.generate_stellar_spectrum(ap)
            f = interp1d([-0.25, 0, 0.25, 0.5], np.array([flux_m03, flux_p0, flux_p03, flux_p05]), kind='linear',
                         axis=0)  # interpolate in alpha space
        elif model == 'marcs' or model == 'MARCS':
            flux_m04 = marcs_m04.generate_stellar_spectrum(ap)
            flux_p0 = marcs_p0.generate_stellar_spectrum(ap)
            flux_p04 = marcs_p04.generate_stellar_spectrum(ap)
            f = interp1d([-0.4, 0, 0.4], np.array([flux_m04, flux_p0, flux_p04]), kind='linear',
                         axis=0)  # interpolate in alpha space
        else:
            raise Exception("Invalid model library.")
        flux_med = f(a) / np.median(f(a))
        return flux_med

    else:
        t, g, z = theta
        ap = (t, g, z)
        if model == 'bosz' or model == 'BOSZ':
            flux = bosz_p0.generate_stellar_spectrum(ap)
        elif model == 'marcs' or model == 'MARCS':
            flux = marcs_p0.generate_stellar_spectrum(ap)
        else:
            raise Exception("Invalid model library.")
        flux_med = flux / np.median(flux)
        return flux_med


# Input variables
alpha = True    # whether you want to include alpha variable models
model = 'BOSZ'  # Either BOSZ or MARCS
theta = [6000, 4, 0, 0]     # the stellar parameters: Teff, log g, [Fe/H] and [alpha/Fe] (if being used)

if alpha == True:
    print('\nGetting models...')
    marcs_m04 = interp_models('marcs_m04')
    marcs_p04 = interp_models('marcs_p04')

    bosz_m03 = interp_models('bosz_m03')
    bosz_p03 = interp_models('bosz_p03')
    bosz_p05 = interp_models('bosz_p05')

marcs_p0 = interp_models('marcs_p0')
bosz_p0 = interp_models('bosz_p0')

# generate the wavelength and model spectrum.
# the wavelength is the same for all models, marcs_p0 instance is used below as an example.

model_wave, model_spec = marcs_p0.model_wave, model_spec(theta, model)

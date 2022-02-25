import os, sys
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
import matplotlib.pyplot as plt

class interp_models:
    """Interpolation of synthetic model spectra"""
    def __init__(self, model_lib, theta):
        self.model_lib = model_lib
        if model_lib == 'bosz_p0':
            self.teff_grid, self.logg_grid, self.met_grid = self.load_grid('bosz_params_all.fits')
            self.model_spec = self.load_spec('bosz_all_mastres.npz')
            self.model_wave = self.load_wave('bosz_all_mastres.npz')
        elif model_lib == 'bosz_p03':
            self.teff_grid, self.logg_grid, self.met_grid = self.load_grid('bosz_params-alpha-p03.fits')
            self.model_spec = self.load_spec('bosz_flux-alpha-p03.npz')
            self.model_wave = self.load_wave('bosz_flux-alpha-p03.npz')
        elif model_lib == 'bosz_m03':
            self.teff_grid, self.logg_grid, self.met_grid = self.load_grid('bosz_params-alpha-m03.fits')
            self.model_spec = self.load_spec('bosz_flux-alpha-m03.npz')
            self.model_wave = self.load_wave('bosz_flux-alpha-m03.npz')
        elif model_lib == 'bosz_p05':
            self.teff_grid, self.logg_grid, self.met_grid = self.load_grid('bosz_params-alpha-p05.fits')
            self.model_spec = self.load_spec('bosz_flux-alpha-p05.npz')
            self.model_wave = self.load_wave('bosz_flux-alpha-p05.npz')
        else:
            raise Exception("Model library not in the directory.")

    def load_grid(self, grid_file):
        """Load the parameters of the selected grid."""
        data = fits.open('libs/' + grid_file)[1].data
        return data['TEFF'], data['LOGG'], data['LOGZ']

    def load_spec(self, spec_file):
        return np.load('libs'+spec_file)['arr_1']

    def load_wave(self, spec_file):
        return np.load('libs' + spec_file)['arr_0'][0]

    def find_nodes(self):
        """Find the nodes which are below and above the input parameters. i.e. the closest parameters on the grid."""
        teff_in, logg_in, met_in = self.theta
        mask_high = np.where((self.teff_grid > teff_in) & (self.logg_grid > logg_in) & (self.met_grid > met_in))
        mask_low = np.where((self.teff_grid < teff_in) & (self.logg_grid < logg_in) & (self.met_grid < met_in))

        teff_high = np.argsort(self.teff_grid[mask_high])[0]
        teff_low = np.argsort(self.teff_grid[mask_low][-1])

        logg_high = np.argsort(self.teff_grid[mask_high][0])
        logg_low = np.argsort(self.teff_grid[mask_low][-1])

        met_high = np.argsort(self.teff_grid[mask_high][0])
        met_low = np.argsort(self.teff_grid[mask_low][-1])

        self.theta_low = [self.teff_grid[mask_low][teff_low], self.logg_grid[mask_low][logg_low],
                     self.met_grid[mask_low][met_low]]
        self.theta_high = [self.teff_grid[mask_high][teff_high], self.logg_grid[mask_high][logg_high],
                     self.met_grid[mask_high][met_high]]



    def get_interp_spec(self):
        #





t1 = np.array([1,2,3,4,5,6,7,8,9,10])
t2 = t1+5

f = interp1d(t1, t2)
plt.plot(t1)
plt.plot(t2)
plt.plot(f(np.arange(len(t1))))
plt.show()

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator
from astropy.io import fits

class interp_models:
    """Interpolation of synthetic model spectra"""
    def __init__(self, model_lib):
        self.model_lib = model_lib
        # BOSZ models
        if model_lib == 'bosz_p0':
            self.teff_grid, self.logg_grid, self.met_grid = self.load_grid('bosz_params_all.fits')
            self.grid_all = np.array([self.teff_grid, self.logg_grid, self.met_grid]).T
            self.model_spec = self.load_spec('bosz_all_mastres.npz')
            self.model_wave = self.load_wave('bosz_all_mastres.npz')
            self.func = LinearNDInterpolator(self.grid_all, self.model_spec)
        elif model_lib == 'bosz_p03':
            self.teff_grid, self.logg_grid, self.met_grid = self.load_grid('bosz_params-alpha-p03.fits')
            self.grid_all = np.array([self.teff_grid, self.logg_grid, self.met_grid]).T
            self.model_spec = self.load_spec('bosz_flux-alpha-p03.npz')
            self.model_wave = self.load_wave('bosz_flux-alpha-p03.npz')
            self.func = LinearNDInterpolator(self.grid_all, self.model_spec)
        elif model_lib == 'bosz_m03':
            self.teff_grid, self.logg_grid, self.met_grid = self.load_grid('bosz_params-alpha-m03.fits')
            self.grid_all = np.array([self.teff_grid, self.logg_grid, self.met_grid]).T
            self.model_spec = self.load_spec('bosz_flux-alpha-m03.npz')
            self.model_wave = self.load_wave('bosz_flux-alpha-m03.npz')
            self.func = LinearNDInterpolator(self.grid_all, self.model_spec)
        elif model_lib == 'bosz_p05':
            self.teff_grid, self.logg_grid, self.met_grid = self.load_grid('bosz_params-alpha-p05.fits')
            self.grid_all = np.array([self.teff_grid, self.logg_grid, self.met_grid]).T
            self.model_spec = self.load_spec('bosz_flux-alpha-p05.npz')
            self.model_wave = self.load_wave('bosz_flux-alpha-p05.npz')
            self.func = LinearNDInterpolator(self.grid_all, self.model_spec)

        # MARCS models
        elif model_lib == 'marcs_m04':
            self.teff_grid, self.logg_grid, self.met_grid = self.load_grid('marcs_params_all-alpha-m04.fits')
            self.grid_all = np.array([self.teff_grid, self.logg_grid, self.met_grid]).T
            self.model_spec = self.load_spec('marcs_all_mastres-alpha-m04.npz')
            self.model_wave = self.load_wave('marcs_all_mastres-alpha-m04.npz')
            self.func = LinearNDInterpolator(self.grid_all, self.model_spec)
        elif model_lib == 'marcs_p0':
            self.teff_grid, self.logg_grid, self.met_grid = self.load_grid('marcs_params_all-alpha0.fits')
            self.grid_all = np.array([self.teff_grid, self.logg_grid, self.met_grid]).T
            self.model_spec = self.load_spec('marcs_all_mastres-alpha0.npz')
            self.model_wave = self.load_wave('marcs_all_mastres-alpha0.npz')
            self.func = LinearNDInterpolator(self.grid_all, self.model_spec)
        elif model_lib == 'marcs_p04':
            self.teff_grid, self.logg_grid, self.met_grid = self.load_grid('marcs_params_all-alpha-p04-v2.fits')
            self.grid_all = np.array([self.teff_grid, self.logg_grid, self.met_grid]).T
            self.model_spec = self.load_spec('marcs_all_mastres-alpha-p04-v2.npz')
            self.model_wave = self.load_wave('marcs_all_mastres-alpha-p04-v2.npz')
            self.func = LinearNDInterpolator(self.grid_all, self.model_spec)
        else:
            raise Exception("Model library not in the directory.")

    def load_grid(self, grid_file):
        """Load the parameters of the selected grid."""
        data = fits.open('pystellibs_SPD/libs/' + grid_file)[1].data
        return data['TEFF'], data['LOGG'], data['LOGZ']

    def load_spec(self, spec_file):
        return np.load('pystellibs_SPD/libs/'+spec_file)['arr_1']

    def load_wave(self, spec_file):
        return np.load('pystellibs_SPD/libs/' + spec_file)['arr_0']

    @staticmethod
    def get_spec(theta, func):
        spec_interp = func(theta)
        if np.isnan(spec_interp).all():
            spec_interp = np.full(len(spec_interp), 1)
        return spec_interp

    def generate_stellar_spectrum(self, theta):
        return interp_models.get_spec(theta=theta, func=self.func)




    def find_nodes(self, theta):
        """Find the nodes which are below and above the input parameters. i.e. the closest parameters on the grid."""
        teff_in, logg_in, met_in = theta
        mask_high = np.where((self.teff_grid > teff_in) & (self.logg_grid > logg_in) & (self.met_grid > met_in))
        mask_low = np.where((self.teff_grid < teff_in) & (self.logg_grid < logg_in) & (self.met_grid < met_in))

        teff_high = np.argsort(self.teff_grid[mask_high])[0]
        teff_low = np.argsort(self.teff_grid[mask_low])[-1]

        logg_high = np.argsort(self.teff_grid[mask_high])[0]
        logg_low = np.argsort(self.teff_grid[mask_low])[-1]

        met_high = np.argsort(self.teff_grid[mask_high])[0]
        met_low = np.argsort(self.teff_grid[mask_low])[-1]

        self.theta_low = [self.teff_grid[mask_low][teff_low], self.logg_grid[mask_low][logg_low],
                     self.met_grid[mask_low][met_low]]
        self.theta_high = [self.teff_grid[mask_high][teff_high], self.logg_grid[mask_high][logg_high],
                     self.met_grid[mask_high][met_high]]

        self.spec_low = self.model_spec[np.where((self.teff_grid == self.theta_low[0]) &
                                                 (self.logg_grid == self.theta_low[1]) &
                                                 (self.met_grid == self.theta_low[2]))][0]
        self.spec_high = self.model_spec[np.where((self.teff_grid == self.theta_high[0]) &
                                                 (self.logg_grid == self.theta_high[1]) &
                                                 (self.met_grid == self.theta_high[2]))][0]


    def spec_between_nodes(self, theta):
        """Given a parameter combination, return an interpolated spctrum for a fixed alpha enhancement."""
        self.find_nodes(theta)
        f = interp1d(self.spec_low, self.spec_high)
        self.spec_interp = f(np.arange(len(self.spec_low)))
        return self.model_wave, self.spec_interp

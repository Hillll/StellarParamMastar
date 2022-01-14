import sys

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


from pystellibs import Marcs_p0, Bosz
import numpy as np
import pandas as pd
import random
import time
from math import pi
from spd_setup import spd_setup
from astropy.io import fits
from scipy.interpolate import interp1d
# from firefly_dust import reddening_fm
from ppxf.ppxf import ppxf
import emcee
import multiprocessing
from multiprocessing import Pool
# multiprocessing.set_start_method('fork')
# from schwimmbad import MultiPool
from corner import corner

var = spd_setup()

marcs_mods = Marcs_p0()
bosz_mods = Bosz()

def model_wave(theta, model):  # model given a set of parameters (theta)
    t, g, z = theta
    ap = (np.log10(t), g, 0, (10 ** (z)) * 0.02)  # try different normalisation
    if model == 'bosz' or model == 'BOSZ':
        wave = np.asarray(bosz_mods.wavelength)
    elif model == 'marcs' or model == 'MARCS':
        wave = np.asarray(marcs_mods.wavelength)
    else:
        raise Exception("Invalid model library.")
    return wave


def model_spec(theta, model):  # model given a set of parameters (theta)
    t, g, z = theta
    ap = (np.log10(t), g, 0, (10 ** (z)) * 0.02)  # try different normalisation
    if model == 'bosz' or model == 'BOSZ':
        flux = np.asarray(bosz_mods.generate_stellar_spectrum(*ap))
    elif model == 'marcs' or model == 'MARCS':
        flux = np.asarray(marcs_mods.generate_stellar_spectrum(*ap))
    else:
        raise Exception("Invalid model library.")
    flux_med = flux / np.median(flux)
    return flux_med


'''

marcs_mods = synth_models('marcs')
bosz_mods = model_spec('bosz')
>>>>>>> 66c395c... update.

class model_spec():
    """Load either MARCS or BOSZ model atmospheres."""

    def __init__(self, model):
        self.model = model
        self.marcs_mods = Marcs_p0()
        self.bosz_mods = Bosz()

    def model_wave(self, theta):  # model given a set of parameters (theta)
        t, g, z = theta
        ap = (np.log10(t), g, 0, (10 ** (z)) * 0.02)  # try different normalisation
        if self.model == 'bosz' or self.model == 'BOSZ':
            wave = np.asarray(self.bosz_mods.wavelength)
        elif self.model == 'marcs' or self.model == 'MARCS':
            wave = np.asarray(self.marcs_mods.wavelength)
        else:
            raise Exception("Invalid model library.")
        return wave

    def model_spec(self, theta):  # model given a set of parameters (theta)
        t, g, z = theta
        ap = (np.log10(t), g, 0, (10 ** (z)) * 0.02)  # try different normalisation
        if self.model == 'bosz' or self.model == 'BOSZ':
            flux = np.asarray(self.bosz_mods.generate_stellar_spectrum(*ap))
        elif self.model == 'marcs' or self.model == 'MARCS':
            flux = np.asarray(self.marcs_mods.generate_stellar_spectrum(*ap))
        else:
            raise Exception("Invalid model library.")
        flux_med = flux / np.median(flux)
        return flux_med
'''

class load_data:
    """Load mastar spectra and the estimates file from Gaia photometry."""

    def __init__(self, data_direc=var.data_direc, spec_file=var.spec_file, est_file=var.est_file,
                 nums_file=var.nums_file):
        self.data_direc = data_direc
        self.spec_file = spec_file
        self.est_file = est_file
        self.nums_file = nums_file

    def get_mastar(self, start=0, end=1000):
        header = fits.open(self.data_direc + self.spec_file)
        self.mangaid = header[1].data['mangaid']
        self.plate_1, self.ifu_1, self.mjd_1 = header[1].data['plate'][start:end + 1], header[1].data['ifudesign'][
                                                                                       start:end + 1], \
                                               header[1].data['mjd'][start:end + 1]
        # self.ifu_1 = np.asarray([int(i) for i in self.ifu_1])   # ensure ifu is int
        self.ra, self.dec = header[1].data['ra'][start:end + 1], header[1].data['dec'][start:end + 1]
        self.wave, self.flux = header[1].data['wave'][0][9:-8], header[1].data['flux'][start:end + 1]
        self.ivar, self.exptime = header[1].data['ivar'][start:end + 1], header[1].data['exptime'][start:end + 1]
        header.close()

    def get_estimates(self, start=0, end=1000):
        # if not isinstance(plate, array)
        header = fits.open(self.data_direc + self.est_file)
        self.meta_data = header[1].data[start:end + 1]
        header.close()

        # apply some conditions to estimates
        for i in range(len(self.meta_data)):
            if self.meta_data['minTEFF_gaia'][i] > self.meta_data['teff_gaia'][i] * 0.84:  # ensure prior is of a certain minimum width
                self.meta_data['minTEFF_gaia'][i] = self.meta_data['teff_gaia'][i] * 0.84
            if self.meta_data['maxTEFF_gaia'][i] < self.meta_data['teff_gaia'][i] * 1.16:
                self.meta_data['maxTEFF_gaia'][i] = self.meta_data['teff_gaia'][i] * 1.16
            if self.meta_data['minTEFF_gaia'][i] < 2500:  # restrict prior range to models +/- some leeway
                self.meta_data['minTEFF_gaia'][i] = 2000
            if self.meta_data['maxTEFF_gaia'][i] > 35000:
                self.meta_data['maxTEFF_gaia'][i] = 40000
            if self.meta_data['minLOGG_gaia'][i] < -0.5:
                self.meta_data['minLOGG_gaia'][i] = -1
            if self.meta_data['maxLOGG_gaia'][i] > 5.5:
                self.meta_data['maxLOGG_gaia'][i] = 6

    def get_targets(self, number):
        file = np.load(self.data_direc + self.nums_file)
        return file['arr_0'][number]


class prepare_spectrum:
    """deal with dead pixels in data, correct for reddening and deal with no gaia info"""

    def __init__(self, wave, flux, ivar, ebv, spec_id):
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.ebv = ebv
        self.spec_id = spec_id

    def dered(self, flux_to_dered, a_v=None, r_v=3.1, model='f99'):
        """ ** Adapted from firefly_dust.py, reddening_fm **
        Determines a Fitzpatrick & Massa reddening curve and returns the de-reddened flux.


        Parameters
        ----------
        wave: ~numpy.ndarray
            wavelength in Angstroms
        flux: ~numpy.ndarray
            mastrum spectrum
        ebv: float
            E(B-V) differential extinction; specify either this or a_v.
        a_v: float
            A(V) extinction; specify either this or ebv.
        r_v: float, optional
            defaults to standard Milky Way average of 3.1
        model: {'f99', 'fm07'}, optional
            * 'f99' is the default Fitzpatrick (1999) [1]_
            * 'fm07' is Fitzpatrick & Massa (2007) [2]_. Currently not R dependent.

        Returns
        -------
        reddening_curve: ~numpy.ndarray
            Multiply to deredden flux, divide to redden.

        Notes
        -----
        Uses Fitzpatrick (1999) [1]_ by default, which relies on the UV
        parametrization of Fitzpatrick & Massa (1990) [2]_ and spline fitting in the
        optical and IR. This function is defined from 910 A to 6 microns, but note
        the claimed validity goes down only to 1150 A. The optical spline points are
        not taken from F99 Table 4, but rather updated versions from E. Fitzpatrick
        (this matches the Goddard IDL astrolib routine FM_UNRED).

        The fm07 model uses the Fitzpatrick & Massa (2007) [3]_ parametrization,
        which has a slightly different functional form. That paper claims it
        preferable, although it is unclear if signficantly (Gordon et al. 2009)
        [4]_. It is not the literature standard, so not default here.

        References
        ----------
        [1] Fitzpatrick, E. L. 1999, PASP, 111, 63
        [2] Fitpatrick, E. L. & Massa, D. 1990, ApJS, 72, 163
        [3] Fitpatrick, E. L. & Massa, D. 2007, ApJ, 663, 320
        [4] Gordon, K. D., Cartledge, S., & Clayton, G. C. 2009, ApJ, 705, 1320

        """

        model = model.lower()
        if model not in ['f99', 'fm07']:
            raise ValueError('model must be f99 or fm07')
        if (a_v is None) and (self.ebv is None):
            raise ValueError('Must specify either a_v or ebv')
        if (a_v is not None) and (self.ebv is not None):
            raise ValueError('Cannot specify both a_v and ebv')
        if a_v is not None:
            ebv = a_v / r_v

        if model == 'fm07':
            raise ValueError('TEMPORARY: fm07 currently not properly R dependent')

        x = 1e4 / self.wave  # inverse microns
        k = np.zeros(x.size)

        if any(x < 0.167) or any(x > 11):
            raise ValueError('fm_dered valid only for wavelengths from 910 A to ' +
                             '6 microns')

        # UV region
        uvsplit = 10000. / 2700.  # Turn 2700A split into inverse microns.
        uv_region = (x >= uvsplit)
        y = x[uv_region]
        k_uv = np.zeros(y.size)

        # Fitzpatrick (1999) model
        if model == 'f99':
            x0, gamma = 4.596, 0.99
            c3, c4 = 3.23, 0.41
            c2 = -0.824 + 4.717 / r_v
            c1 = 2.030 - 3.007 * c2
            D = y ** 2 / ((y ** 2 - x0 ** 2) ** 2 + y ** 2 * gamma ** 2)
            F = np.zeros(y.size)
            valid = (y >= 5.9)
            F[valid] = 0.5392 * (y[valid] - 5.9) ** 2 + 0.05644 * (y[valid] - 5.9) ** 3
            k_uv = c1 + c2 * y + c3 * D + c4 * F
        # Fitzpatrick & Massa (2007) model
        if model == 'fm07':
            x0, gamma = 4.592, 0.922
            c1, c2, c3, c4, c5 = -0.175, 0.807, 2.991, 0.319, 6.097
            D = y ** 2 / ((y ** 2 - x0 ** 2) ** 2 + y ** 2 * gamma ** 2)
            valid = (y <= c5)
            k_uv[valid] = c1 + c2 * y[valid] + c3 * D[valid]
            valid = (y > c5)
            k_uv[valid] = c1 + c2 * y[valid] + c3 * D[valid] + c4 * (y[valid] - c5) ** 2

        k[uv_region] = k_uv

        # Calculate values for UV spline points to anchor OIR fit
        x_uv_spline = 10000. / np.array([2700., 2600.])
        D = x_uv_spline ** 2 / ((x_uv_spline ** 2 - x0 ** 2) ** 2 + x_uv_spline ** 2 * gamma ** 2)
        k_uv_spline = c1 + c2 * x_uv_spline + c3 * D

        # Optical / IR
        OIR_region = (x < uvsplit)
        y = x[OIR_region]
        k_OIR = np.zeros(y.size)

        # Fitzpatrick (1999) model
        if model == 'f99':
            # The OIR anchors are up from IDL astrolib, not F99.
            anchors_extinction = np.array([0, 0.26469 * r_v / 3.1, 0.82925 * r_v / 3.1,  # IR
                                           -0.422809 + 1.00270 * r_v + 2.13572e-04 * r_v ** 2,  # optical
                                           -5.13540e-02 + 1.00216 * r_v - 7.35778e-05 * r_v ** 2,
                                           0.700127 + 1.00184 * r_v - 3.32598e-05 * r_v ** 2,
                                           (1.19456 + 1.01707 * r_v - 5.46959e-03 * r_v ** 2 + 7.97809e-04 * r_v ** 3 +
                                            -4.45636e-05 * r_v ** 4)])
            anchors_k = np.append(anchors_extinction - r_v, k_uv_spline)
            # Note that interp1d requires that the input abscissa is monotonically
            # _increasing_. This is opposite the usual ordering of a spectrum, but
            # fortunately the _output_ abscissa does not have the same requirement.
            anchors_x = 1e4 / np.array([26500., 12200., 6000., 5470., 4670., 4110.])
            anchors_x = np.append(0., anchors_x)  # For well-behaved spline.
            anchors_x = np.append(anchors_x, x_uv_spline)
            OIR_spline = interp1d(anchors_x, anchors_k, kind='cubic')
            k_OIR = OIR_spline(y)
        # Fitzpatrick & Massa (2007) model
        if model == 'fm07':
            anchors_k_opt = np.array([0., 1.322, 2.055])
            IR_wave = np.array([float('inf'), 4., 2., 1.333, 1.])
            anchors_k_IR = (-0.83 + 0.63 * r_v) * IR_wave ** -1.84 - r_v
            anchors_k = np.append(anchors_k_IR, anchors_k_opt)
            anchors_k = np.append(anchors_k, k_uv_spline)
            anchors_x = np.array([0., 0.25, 0.50, 0.75, 1.])  # IR
            opt_x = 1e4 / np.array([5530., 4000., 3300.])  # optical
            anchors_x = np.append(anchors_x, opt_x)
            anchors_x = np.append(anchors_x, x_uv_spline)
            OIR_spline = interp1d(anchors_x, anchors_k, kind='cubic')
            k_OIR = OIR_spline(y)

        k[OIR_region] = k_OIR

        reddening_curve = 10 ** (0.4 * self.ebv * (k + r_v))

        corrected_flux = flux_to_dered * reddening_curve

        return corrected_flux

    def get_med_data(self):
        """ Median normalise the data, we also need to calculate the error as a percentage of the flux.
                If dead pixels occur in the spectrum/error array, these need to be interpolated first."""

        if np.all(self.flux > 0) == False:  # deal with dead pixels
            flux_x = np.arange(len(self.flux))
            idxx = np.where(self.flux > 0)
            ff = interp1d(flux_x[idxx], self.flux[idxx],
                          fill_value='extrapolate')  # interp function with non zero values
            flux_new = ff(flux_x)  # interpolate where zero values occur

            corrected_flux = self.dered(flux_to_dered=flux_new)  # call the dered fn that returns de-reddened spectrum

            ivar_ = self.ivar[9:-8]  # get inverse variance
            ivar_x = np.arange(len(ivar_))
            idx = np.nonzero(ivar_)  # nonzero values in ivar
            f = interp1d(ivar_x[idx], ivar_[idx], fill_value='extrapolate')  # interp function with non zero values
            ivar_new = f(ivar_x)  # interpolate where zero values occur
            sd = ivar_new ** -0.5  # change inverse variance to standard error
            sd_pcnt = sd / flux_new  # error as a percentage of the flux

            self.corrected_flux_med = corrected_flux / np.median(corrected_flux)  # median normalise
            self.yerr = self.corrected_flux_med * sd_pcnt
        else:
            ivar_ = self.ivar[9:-8]  # get inverse variance
            ivar_x = np.arange(len(ivar_))
            idx = np.nonzero(ivar_)  # nonzero values in ivar
            f = interp1d(ivar_x[idx], ivar_[idx], fill_value='extrapolate')  # interp function with non zero values
            ivar_new = f(ivar_x)  # interpolate where zero values occur
            sd = ivar_new ** -0.5  # change inverse variance to standard error
            sd_pcnt = sd / self.flux  # error as a percentage of the flux

            corrected_flux = self.dered(flux_to_dered=self.flux)  # call the dered fn that returns de-reddened spectrum

            self.corrected_flux_med = corrected_flux / np.median(corrected_flux)  # median normalise
            self.yerr = self.corrected_flux_med * sd_pcnt

    def catch_remaining(self, meta_data):
        if np.any(np.isnan(self.corrected_flux_med)) == True:
            print('Invalid data, skipping to next object.')
            return True
        elif self.ebv < 0:
            print('Invalid E(B-V), skipping to next object.')
            return True
        elif meta_data['teff_gaia'] < -10 or meta_data['LOGG_gaia'] < -10 or meta_data['ZH_gaia'] < -10:
            print('Invalid priors, skipping to next object.')
            return True
        elif meta_data['teff_gaia'] > 40000 or meta_data['teff_gaia'] < 2000:
            print('Invalid priors, skipping to next object.')
            return True
        else:
            return False

class toy_func:
    """The functions required for running emcee."""

    # marcs_mods = synth_models('marcs')
    # bosz_mods = synth_models('bosz')

    def __init__(self, model, flux, yerr, meta_data, parallel):
        self.model = model
        self.flux = flux
        self.yerr = yerr
        self.meta_data = meta_data
        # self.marcs_mods = marcs_mods
        # self.bosz_mods = bosz_mods
        self.parallel = parallel


    def starting(self):
        p0_ = []  # generate random starting points for the walkers, uniform across min and max priors
        for j in range(var.nwalkers):
            temp = [random.uniform(self.meta_data['minTEFF_gaia'], self.meta_data['maxTEFF_gaia']),
                    random.uniform(self.meta_data['minLOGG_gaia'], self.meta_data['maxLOGG_gaia']),
                    random.uniform(-2.5, 0.5)]
            p0_.append(temp)
        self.p0_ = np.asarray(p0_)

    def sample(self):
        print('\nRunning MCMC')
        self.sampler = self.main()
        return self.sampler.flatchain

    # @staticmethod
    def main(self):  # The MCMC routine
        if self.parallel == True:
            np.random.seed(42)
            initial = np.random.randn(32, 5)
            nwalkers, ndim = initial.shape
            nsteps = 100
            with Pool() as pool:
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=[self.flux, self.yerr],
                                                    pool=pool)
                    start = time.time()
                    sampler.run_mcmc(initial, nsteps, progress=True)
                    end = time.time()
                    multi_time = end - start
                    print("Multiprocessing took {0:.1f} seconds".format(multi_time))
                    return sampler

    @staticmethod
    def lnprob(theta, flux, yerr):  # posterior probability bosz
        t0 = time.time()
        t = time.time() + 5
        #print(toy_func.lnprior2(theta))
        print(len(flux), len(yerr))
        while True:
            if time.time() >= t:
                break
        print('Time on cpu {}: {}'.format(multiprocessing.current_process().name, time.time()))
        return -0.5 * np.sum(theta ** 2)

    @staticmethod
    def lnprior2(theta):
        spec = model_spec([5000, 3, 0], 'BOSZ')
        return len(spec)

class mcmc_v2():
    """The functions required for running emcee."""

    def __init__(self, model, flux, yerr, meta_data, parallel=True):
        self.model = model
        self.flux = flux
        self.yerr = yerr
        self.meta_data = meta_data
        self.parallel = parallel
        print('\nUsing {} CPUs'.format(multiprocessing.cpu_count()))

    def starting(self):
        p0_ = []  # generate random starting points for the walkers, uniform across min and max priors
        for j in range(var.nwalkers):
            temp = [random.uniform(self.meta_data['minTEFF_gaia'], self.meta_data['maxTEFF_gaia']),
                    random.uniform(self.meta_data['minLOGG_gaia'], self.meta_data['maxLOGG_gaia']),
                    random.uniform(-2.5, 0.5)]
            p0_.append(temp)
        self.p0_ = np.asarray(p0_)

    def sample(self):
        print('\nRunning MCMC using {} models...'.format(self.model))
        self.sampler = self.main()
        return self.sampler

    def main(self):  # The MCMC routine
        if self.parallel == True:
            with Pool() as pool:
                print(np.show_config())
                sampler = emcee.EnsembleSampler(var.nwalkers, var.ndim, self.lnprob, a=var.a, pool=pool,
                                                args=[self.model, self.flux, self.yerr, self.meta_data])
                # Burn in
                print(self.p0_, var.burnin)
                p0_, _, _ = sampler.run_mcmc(self.p0_, var.burnin,
                                             progress=True)  # this diminishes the influence of starting values
                print('\nFinished burn in.')
                # Production
                sampler.run_mcmc(p0_, var.niter, progress=True)
                print('\nFinished {} iterations'.format(var.niter))
                return sampler
        elif self.parallel == False:
            sampler = emcee.EnsembleSampler(var.nwalkers, var.ndim, self.lnprob, a=var.a)
            # Burn in
            print(self.p0_, var.burnin)
            p0_, _, _ = sampler.run_mcmc(self.p0_, var.burnin,
                                         progress=True)  # this diminishes the influence of starting values
            print('\nFinished burn in.')
            # Production
            sampler.run_mcmc(p0_, var.niter, progress=True)
            print('\nFinished {} iterations'.format(var.niter))
            return sampler

    @staticmethod
    def lnprob(theta, model, flux, yerr, meta_data):  # posterior probability bosz
        t0 = time.time()
        lp = mcmc_v2.lnprior(theta, model, meta_data)
        if not np.isfinite(lp):
            return -np.inf
        temp = lp + mcmc_v2.lnlike(theta, model, flux, yerr)
        print('Time on cpu {}: {}'.format(multiprocessing.current_process().name, time.time() - t0))
        return temp

    @staticmethod
    def lnprior(theta, model, meta_data):  # prior estimate of the data - flat
        t, g, z = theta
        if model == 'marcs' or model == 'MARCS':
            if meta_data['minTEFF_gaia'] <= t <= meta_data['maxTEFF_gaia'] and \
                    meta_data['minLOGG_gaia'] <= g <= meta_data['maxLOGG_gaia'] and \
                    t > 2000 and -3 <= z <= 1:
                return 1
            else:
                return -np.inf

        elif model == 'bosz' or model == 'BOSZ':
            if meta_data['minTEFF_gaia'] <= t <= meta_data['maxTEFF_gaia'] and \
                    meta_data['minLOGG_gaia'] <= g <= meta_data['maxLOGG_gaia'] and \
                    t > 3500 and -3 <= z <= 0.5:
                return 1
            else:
                return -np.inf

    @staticmethod
    def lnlike(theta, model, flux, yerr):  # likelihood fn that evaluates best fit
        if model == 'bosz' or model == 'BOSZ':
            model_flux = np.asarray(model_spec(theta, model))  # model flux with mcmc proposed params (theta)
        elif model == 'marcs' or model == 'MARCS':
            model_flux = np.asarray(model_spec(theta, model))  # model flux with mcmc proposed parameters (theta)

        if np.array_equal(model_flux,
                          np.ones(len(model_flux))):  # if theta is outside model grid bbox, array of ones is returned
            return -np.inf  # return -inf so walkers move away form this parameter combination
        elif np.isnan(model_flux).any():
            return -np.inf  # return -inf so walkers move away form this parameter combination
        else:
            sol = (ppxf(model_flux, flux, noise=yerr, velscale=var.velscale, start=var.start, degree=-1,
                        mdegree=var.mdegree, moments=var.moments,
                        quiet=True))  # run ppxf with interpolated model

            t = ((flux - sol.bestfit) ** 2) / (yerr ** 2)
            mask = ~np.isinf(t) & ~np.isnan(t)  # remove inf values from array
            LnLike = -0.5 * np.sum(np.log(2 * pi * (yerr ** 2)) + t[mask]) / len(yerr)
            return LnLike



'''
class plotting:
    

def chi2_lh(data, model, error):
    chi = np.sum(((data - model) ** 2 / error ** 2))
    return chi


def max_post(samp):
    X = np.linspace(min(samp), max(samp))
    func = scipy.stats.gaussian_kde(samp)
    return X[np.argmax(func(X))]'''

import sys

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pystellibs_SPD import Marcs_p0, Bosz
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import time
import arviz, scipy
from math import pi
from spd_setup import spd_setup
from astropy.io import fits
from scipy.interpolate import interp1d
# from firefly_dust import reddening_fm
from ppxf.ppxf import ppxf
import emcee
import multiprocessing
from multiprocessing import Pool
from corner import corner

var = spd_setup()
if var.alpha:
    print('\nGetting models...')
    from pystellibs_SPD import Marcs_m04, Marcs_p04
    from pystellibs_SPD import Bosz_p03, Bosz_p05, Bosz_m03

    marcs_m04 = Marcs_m04()
    marcs_p04 = Marcs_p04()

    bosz_m03 = Bosz_m03()
    bosz_p03 = Bosz_p03()
    bosz_p05 = Bosz_p05()

marcs_p0 = Marcs_p0()
bosz_p0 = Bosz()


def model_spec(theta, model):  # model given a set of parameters (theta)
    if var.alpha:
        t, g, z, a = theta
        ap = (np.log10(t), g, 0, (10 ** (z)) * 0.02)  # try different normalisation
        if model == 'bosz' or model == 'BOSZ':
            flux_m03 = bosz_m03.generate_stellar_spectrum(*ap)  # get alpha model for each combination of t,g,z
            flux_p0 = bosz_p0.generate_stellar_spectrum(*ap)
            flux_p03 = bosz_p03.generate_stellar_spectrum(*ap)
            flux_p05 = bosz_p05.generate_stellar_spectrum(*ap)
            f = interp1d([-0.25, 0, 0.25, 0.5], np.array([flux_m03, flux_p0, flux_p03, flux_p05]), kind='linear',
                         axis=0)  # interpolate in alpha space
        elif model == 'marcs' or model == 'MARCS':
            flux_m04 = marcs_m04.generate_stellar_spectrum(*ap)
            flux_p0 = marcs_p0.generate_stellar_spectrum(*ap)
            flux_p04 = marcs_p04.generate_stellar_spectrum(*ap)
            f = interp1d([-0.4, 0, 0.4], np.array([flux_m04, flux_p0, flux_p04]), kind='linear',
                         axis=0)  # interpolate in alpha space
            flux = np.asarray(marcs_p0.generate_stellar_spectrum(*ap))
        else:
            raise Exception("Invalid model library.")
        flux_med = f(a) / np.median(f(a))
        return flux_med

    else:
        t, g, z = theta
        ap = (np.log10(t), g, 0, (10 ** (z)) * 0.02)  # try different normalisation
        if model == 'bosz' or model == 'BOSZ':
            flux = np.asarray(bosz_p0.generate_stellar_spectrum(*ap))
        elif model == 'marcs' or model == 'MARCS':
            flux = np.asarray(marcs_p0.generate_stellar_spectrum(*ap))
        else:
            raise Exception("Invalid model library.")
        flux_med = flux / np.median(flux)
        return flux_med


class load_data:
    """Load spectra and the estimates file from Gaia photometry."""

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
        self.ifu_1 = np.asarray([int(i) for i in self.ifu_1])   # ensure ifu is int
        self.ra, self.dec = header[1].data['ra'][start:end + 1], header[1].data['dec'][start:end + 1]
        self.wave, self.flux = header[1].data['wave'][0][9:-8], header[1].data['flux'][start:end + 1]
        self.ivar, self.exptime = header[1].data['ivar'][start:end + 1], header[1].data['exptime'][start:end + 1]
        header.close()

        self.pim = [int(str((self.plate_1[i])) + str((self.ifu_1[i])) + str((self.mjd_1[i]))) for i in
                    range(len(self.plate_1))]

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

    def get_solar(self):
        self.wave = np.load(var.data_direc + 'SOLAR_spec_mastar-res-nodegrade.npz')['arr_0']
        self.flux = np.load(var.data_direc + 'SOLAR_spec_mastar-res-nodegrade.npz')['arr_1']
        self.yerr = np.load(var.data_direc + 'SOLAR_spec_mastar-res-nodegrade.npz')['arr_2']


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

class prepare_spectrum_solar:
    """deal with dead pixels in data, correct for reddening and deal with no gaia info"""

    def __init__(self, wave, flux, yerr):
        self.wave = wave
        self.flux = flux
        self.yerr = yerr

    def get_med_data(self):
        """ Median normalise the data, we also need to calculate the error as a percentage of the flux.
                If dead pixels occur in the spectrum/error array, these need to be interpolated first."""

        sd_pcnt = self.yerr  # error as a percentage of the flux
        self.corrected_flux_med = self.flux / np.median(self.flux)  # median normalise
        self.yerr = self.corrected_flux_med * sd_pcnt


class mcmc_solar:
    """The functions required for running emcee."""

    def __init__(self, flux, yerr, parallel=True):
        self.sample_all = {}
        self.flux = flux
        self.yerr = yerr
        self.parallel = parallel
        self.gaia_priors = {'minTEFF_gaia':3000, 'maxTEFF_gaia':7000,
                            'minLOGG_gaia':2, 'maxLOGG_gaia':5}
        # self.minTEFF_gaia, self.maxTEFF_gaia = meta_data['minTEFF_gaia'], meta_data['maxTEFF_gaia']
        # self.minLOGG_gaia, self.maxLOGG_gaia = meta_data['minLOGG_gaia'], meta_data['maxLOGG_gaia']
        print('\nUsing {} CPUs'.format(multiprocessing.cpu_count()))

    def starting(self):
        p0_ = []  # generate random starting points for the walkers, uniform across min and max priors
        for j in range(var.nwalkers):
            if var.alpha:
                temp = [random.uniform(self.gaia_priors['minTEFF_gaia'], self.gaia_priors['maxTEFF_gaia']),
                        random.uniform(self.gaia_priors['minLOGG_gaia'], self.gaia_priors['maxLOGG_gaia']),
                        random.uniform(-2.5, 0.5), random.uniform(-0.25, 0.4)]  # within marcs and bosz alpha range
            else:
                temp = [random.uniform(self.gaia_priors['minTEFF_gaia'], self.gaia_priors['maxTEFF_gaia']),
                        random.uniform(self.gaia_priors['minLOGG_gaia'], self.gaia_priors['maxLOGG_gaia']),
                        random.uniform(-2.5, 0.5)]
            p0_.append(temp)
        self.p0_ = np.asarray(p0_)

    def sample(self, model):
        print('\nRunning MCMC using {} models...'.format(model))
        sampler = self.main(model)
        self.sample_all[model] = sampler
        self.sample_all


    def main(self, model):  # The MCMC routine
        if self.parallel == True:
            with Pool() as pool:
                print(np.show_config())
                print(scipy.show_config())
                sampler = emcee.EnsembleSampler(var.nwalkers, var.ndim, self.lnprob, a=var.a, pool=pool,
                                                args=[model, self.flux, self.yerr, self.gaia_priors])
                # Burn in
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
    def lnprob(theta, model, flux, yerr, gaia_priors):  # posterior probability bosz
        t0 = time.time()
        lp = mcmc.lnprior(theta, model, gaia_priors)
        if not np.isfinite(lp):
            return -np.inf
        temp = lp + mcmc.lnlike(theta, model, flux, yerr)
        # print('Time on cpu {}: {}'.format(multiprocessing.current_process().name, time.time() - t0))
        return temp

    @staticmethod
    def lnprior(theta, model, gaia_priors):  # prior estimate of the data - flat
        if var.alpha:
            t, g, z, a = theta
            if model == 'marcs' or model == 'MARCS':
                if gaia_priors['minTEFF_gaia'] <= t <= gaia_priors['maxTEFF_gaia'] and \
                        gaia_priors['minLOGG_gaia'] <= g <= gaia_priors['maxLOGG_gaia'] and \
                        t > 2500 and -2 <= z <= 1 and -0.4 <= a <= 0.4:
                    return 1
                elif gaia_priors['minTEFF_gaia'] <= t <= gaia_priors['maxTEFF_gaia'] and \
                        gaia_priors['minLOGG_gaia'] <= g <= gaia_priors['maxLOGG_gaia'] and \
                        t > 2500 and -2.5 <= z <= 1 and 0 <= a <= 0.4:
                    return 1
                else:
                    return -np.inf

            elif model == 'bosz' or model == 'BOSZ':
                if gaia_priors['minTEFF_gaia'] <= t <= gaia_priors['maxTEFF_gaia'] and \
                        gaia_priors['minLOGG_gaia'] <= g <= gaia_priors['maxLOGG_gaia'] and \
                        t > 3500 and -2.5 <= z <= 0.5 and -0.25 <= a <= 0.5:
                    return 1
                else:
                    return -np.inf
        else:
            t, g, z = theta
            if model == 'marcs' or model == 'MARCS':
                if gaia_priors['minTEFF_gaia'] <= t <= gaia_priors['maxTEFF_gaia'] and \
                        gaia_priors['minLOGG_gaia'] <= g <= gaia_priors['maxLOGG_gaia'] and \
                        t > 2000 and -3 <= z <= 1:
                    return 1
                else:
                    return -np.inf

            elif model == 'bosz' or model == 'BOSZ':
                if gaia_priors['minTEFF_gaia'] <= t <= gaia_priors['maxTEFF_gaia'] and \
                        gaia_priors['minLOGG_gaia'] <= g <= gaia_priors['maxLOGG_gaia'] and \
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

class mcmc:
    """The functions required for running emcee."""

    def __init__(self, flux, yerr, meta_data, parallel=True):
        self.autocorr = []
        self.sample_all = {}
        self.flux = flux
        self.yerr = yerr
        self.meta_data = meta_data
        self.parallel = parallel
        self.gaia_priors = {'minTEFF_gaia':meta_data['minTEFF_gaia'], 'maxTEFF_gaia':meta_data['maxTEFF_gaia'],
                            'minLOGG_gaia':meta_data['minLOGG_gaia'], 'maxLOGG_gaia':meta_data['maxLOGG_gaia']}
        # self.minTEFF_gaia, self.maxTEFF_gaia = meta_data['minTEFF_gaia'], meta_data['maxTEFF_gaia']
        # self.minLOGG_gaia, self.maxLOGG_gaia = meta_data['minLOGG_gaia'], meta_data['maxLOGG_gaia']
        print('\nUsing {} CPUs'.format(multiprocessing.cpu_count()))

    def starting(self):
        p0_ = []  # generate random starting points for the walkers, uniform across min and max priors
        for j in range(var.nwalkers):
            if var.alpha:
                temp = [random.uniform(self.meta_data['minTEFF_gaia'], self.meta_data['maxTEFF_gaia']),
                        random.uniform(self.meta_data['minLOGG_gaia'], self.meta_data['maxLOGG_gaia']),
                        random.uniform(-2.5, 0.5), random.uniform(-0.25, 0.4)]  # within marcs and bosz alpha range
            else:
                temp = [random.uniform(self.meta_data['minTEFF_gaia'], self.meta_data['maxTEFF_gaia']),
                        random.uniform(self.meta_data['minLOGG_gaia'], self.meta_data['maxLOGG_gaia']),
                        random.uniform(-2.5, 0.5)]
            p0_.append(temp)
        self.p0_ = np.asarray(p0_)

    def sample(self, model):
        print('\nRunning MCMC using {} models...'.format(model))
        sampler = self.main(model)
        self.sample_all[model] = sampler
        self.sample_all


    def main(self, model):  # The MCMC routine
        if self.parallel == True:
            with Pool() as pool:
                backend = emcee.backends.HDFBackend(var.output_direc + 'chains.h5')     # backend to save chains
                backend.reset(var.nwalkers, var.ndim)

                sampler = emcee.EnsembleSampler(var.nwalkers, var.ndim, self.lnprob, a=var.a, pool=pool,
                                                args=[model, self.flux, self.yerr, self.gaia_priors], backend=backend)

                # Burn in diminishes the influence of starting values
                p0_, _, _ = sampler.run_mcmc(self.p0_, var.burnin, progress=True)
                print('\nFinished burn in.')

                params_old = np.zeros(4)
                for sample in sampler.sample(p0_, iterations=var.niter, progress=True):
                    # check convergence every N steps
                    if sampler.iteration % 50 == 0:
                        # Compute the autocorrelation time at this iteration
                        tau = sampler.get_autocorr_time(tol=0)
                        self.autocorr.append(np.mean(tau))
                        # check params
                        steady, params_old = self.compare_params(params_old, sampler)
                        # Check convergence
                        converged = np.all(tau * 15 < sampler.iteration) and steady == 1
                        if converged:
                            break
                print('\nFinished in {} iterations.'.format(sampler.iteration))
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
    def compare_params(params_old, sampler_new):
        """Compare the parameters between iterations. Returns 0 of & change between any parameters is > 1 %, returns
        1 otherwise."""
        samples_new_clean = sampler_new.chain.T[:, var.burnin:, :].reshape((var.ndim, -1))
        params_new = np.array([np.median(samples_new_clean[i]) for i in range(var.ndim)])
        pcnt_change = np.array([mcmc.get_pcnt_change(i, j) for i, j in zip(params_old, params_new)])
        print(pcnt_change)
        if np.any(pcnt_change > 1):
            return 0, params_new
        else:
            return 1, params_new

    @staticmethod
    def get_pcnt_change(a, b):
        """Calculate absolute percentage change between two values"""
        return abs((b - a)/a)*100

    @staticmethod
    def lnprob(theta, model, flux, yerr, gaia_priors):  # posterior probability bosz
        t0 = time.time()
        lp = mcmc.lnprior(theta, model, gaia_priors)
        if not np.isfinite(lp):
            return -np.inf
        temp = lp + mcmc.lnlike(theta, model, flux, yerr)
        # print('Time on cpu {}: {}'.format(multiprocessing.current_process().name, time.time() - t0))
        return temp

    @staticmethod
    def lnprior(theta, model, gaia_priors):  # prior estimate of the data - flat
        if var.alpha:
            t, g, z, a = theta
            if model == 'marcs' or model == 'MARCS':
                if gaia_priors['minTEFF_gaia'] <= t <= gaia_priors['maxTEFF_gaia'] and \
                        gaia_priors['minLOGG_gaia'] <= g <= gaia_priors['maxLOGG_gaia'] and \
                        t > 2500 and -2 <= z <= 1 and -0.4 <= a <= 0.4:
                    return 1
                elif gaia_priors['minTEFF_gaia'] <= t <= gaia_priors['maxTEFF_gaia'] and \
                        gaia_priors['minLOGG_gaia'] <= g <= gaia_priors['maxLOGG_gaia'] and \
                        t > 2500 and -2.5 <= z <= 1 and 0 <= a <= 0.4:
                    return 1
                else:
                    return -np.inf

            elif model == 'bosz' or model == 'BOSZ':
                if gaia_priors['minTEFF_gaia'] <= t <= gaia_priors['maxTEFF_gaia'] and \
                        gaia_priors['minLOGG_gaia'] <= g <= gaia_priors['maxLOGG_gaia'] and \
                        t > 3500 and -2.5 <= z <= 0.5 and -0.25 <= a <= 0.5:
                    return 1
                else:
                    return -np.inf
        else:
            t, g, z = theta
            if model == 'marcs' or model == 'MARCS':
                if gaia_priors['minTEFF_gaia'] <= t <= gaia_priors['maxTEFF_gaia'] and \
                        gaia_priors['minLOGG_gaia'] <= g <= gaia_priors['maxLOGG_gaia'] and \
                        t > 2000 and -3 <= z <= 1:
                    return 1
                else:
                    return -np.inf

            elif model == 'bosz' or model == 'BOSZ':
                if gaia_priors['minTEFF_gaia'] <= t <= gaia_priors['maxTEFF_gaia'] and \
                        gaia_priors['minLOGG_gaia'] <= g <= gaia_priors['maxLOGG_gaia'] and \
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


class point_estimates:
    """Use the sampler output to calculate point estimate parameters and their errors."""

    def __init__(self, spec_info, pim):
        self.params = {}
        self.chi = {}
        self.ppxf_fit = {}
        self.pim = pim
        self.alpha = var.alpha

        # data from 'clean_spec' instance.
        self.mast_flux = spec_info.corrected_flux_med
        self.yerr = spec_info.yerr

    def flatchain(self, mcmc_run, model):
        """remove burnin and flatten chain to 1D"""
        samples_flat = mcmc_run.sample_all[model].chain
        self.samples_cut = samples_flat[:, var.burnin:, :]
        self.samples_clean = self.samples_cut.reshape((-1, var.ndim))
        return self.samples_clean

    @staticmethod
    def get_median(chain):
        """Median of the posterior for a given parameters"""
        pcntiles = np.percentile(chain, [16, 50, 84])
        q = np.diff(pcntiles)
        return [pcntiles[1], q[0], q[1]]

    @staticmethod
    def get_mode(chain):
        """Mode of the posterior for a given parameters"""
        X = np.linspace(min(chain), max(chain))
        func = scipy.stats.gaussian_kde(chain)
        mode = X[np.argmax(func(X))]
        # calculate the errors using credible intervals
        mode_err_dn = (mode - arviz.hdi(chain, 0.68)[0])
        mode_err_up = (arviz.hdi(chain, 0.68)[1] - mode)
        return [mode, mode_err_dn, mode_err_up]

    def params_err(self, model, teff='median', logg='mode', zh='mode', alpha='mode'):
        """calculate the median or mode of the distribution, depending on what is required.
        Also returns errors"""
        params_temp = []
        for c, i in enumerate(self.samples_clean[:].T):  # assuming order of params is: teff, logg, met, alpha
            if c == 0 and teff == 'median':
                params_temp.append(self.get_median(i))
            elif c == 0 and teff == 'mode':
                params_temp.append(self.get_mode(i))

            if c == 1 and logg == 'median':
                params_temp.append(self.get_median(i))
            elif c == 1 and logg == 'mode':
                params_temp.append(self.get_mode(i))

            if c == 2 and zh == 'median':
                params_temp.append(self.get_median(i))
            elif c == 2 and zh == 'mode':
                params_temp.append(self.get_mode(i))

            if c == 3 and alpha == 'median':
                params_temp.append(self.get_median(i))
            elif c == 3 and alpha == 'mode':
                params_temp.append(self.get_mode(i))
        params_temp = np.asarray(params_temp)
        self.params[model] = params_temp

    def get_model_fit(self, model):
        """Get polynomial corrected model fit from pPXF."""
        if self.alpha:
            model_flux = model_spec([self.params[model][0][0], self.params[model][1][0], self.params[model][2][0],
                                    self.params[model][3][0]], model)
            sol = (ppxf(model_flux, self.mast_flux, noise=self.yerr, velscale=var.velscale, start=var.start, degree=-1,
                        mdegree=var.mdegree, moments=var.moments, quiet=True))
            bestfit_model = sol.bestfit
            self.ppxf_fit[model] = bestfit_model      # append fit to the ppxf fit dictionary
            return self.ppxf_fit
        else:
            model_flux = model_spec([self.params[model][0][0], self.params[model][1][0], self.params[model][2][0]],
                                    model)
            sol = (ppxf(model_flux, self.mast_flux, noise=self.yerr, velscale=var.velscale, start=var.start, degree=-1,
                        mdegree=var.mdegree, moments=var.moments, quiet=True))
            bestfit_model = sol.bestfit
            self.ppxf_fit[model] = bestfit_model  # append fit to the ppxf fit dictionary
            return self.ppxf_fit

    def get_chi2(self, model):
        """Get chi2 based on the point estimate parameters and append to chi dictionary for later comparison."""
        self.chi[model] = np.sum(((self.mast_flux - self.ppxf_fit[model]) ** 2 / self.yerr ** 2)) / (len(self.mast_flux
                                                                                                         - var.ndim))
        return self.chi

    def save_data(self, model):
        if not os.path.exists(var.output_direc):
            os.makedirs(var.output_direc)
        np.savez(var.output_direc + str(self.pim) + '_' + model + '_params.npz', self.params[model], self.chi[model])
        np.savez(var.output_direc + str(self.pim) + '_' + model + '_chains.npz', self.samples_cut)


class plotting:
    """If var.plot == True, then plot bestfit, corner and trace."""
    def __init__(self, point_estimates, clean_spec, pim, c, model):
        self.model = model
        self.pim = pim
        self.clean_spec = clean_spec
        self.point_estimates = point_estimates
        if var.plot:
            if not os.path.exists(var.plots_output_direc + str(self.pim)):
                os.makedirs(var.plots_output_direc + str((self.pim)))

            plotting.trace(self)    # create trace plot and save
            plotting.bestfit(self)   # create bestfit plot and save
            plotting.corner(self)   # create corner plot and save


    def trace(self):
        """Plot the trace for each parameter."""

        plt.figure(figsize=(16, 30))
        plt.subplot(var.ndim, 1, 1)
        plt.plot(self.point_estimates.samples_cut[:,:,0].T, '--', color='k', alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.ylabel('Effective Temperature (Kelvin)', fontsize=16)
        plt.xlabel('Iterations', fontsize=16)

        plt.subplot(var.ndim, 1, 2)
        plt.plot(self.point_estimates.samples_cut[:,:,1].T, '--', color='k', alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.ylabel('Log g', fontsize=16)
        plt.xlabel('Iterations', fontsize=16)

        plt.subplot(var.ndim, 1, 3)
        plt.plot(self.point_estimates.samples_cut[:,:,2].T, '--', color='k', alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.ylabel('Metallicity ([Fe/H])', fontsize=16)
        plt.xlabel('Iterations', fontsize=16)

        if var.alpha:
            plt.subplot(var.ndim, 1, 4)
            plt.plot(self.point_estimates.samples_cut[:,:,3].T, '--', color='k', alpha=0.3)
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.ylabel(r'Alpha abundance $([\alpha/Fe])$', fontsize=16)
            plt.xlabel('Iterations', fontsize=16)

        plt.tight_layout()
        plt.savefig(var.plots_output_direc + str(self.pim) + '/' + self.model + '_trace.png', bbox_inches='tight')
        plt.close()

    def bestfit(self):
        """Plot bestfit for model spec"""
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 12), sharey=False, sharex=True)
        axes.set_title('MaStar PIM: ' + str(self.pim), fontsize=32)
        divider = make_axes_locatable(axes)
        ax2 = divider.append_axes("bottom", size="36%", pad=0)
        axes.figure.add_axes(ax2)
        axes.plot(self.clean_spec.wave, self.clean_spec.corrected_flux_med, 'k', linewidth=3, label='MaStar spectrum')
        axes.plot(self.clean_spec.wave, self.point_estimates.ppxf_fit['BOSZ'], c='b', lw=2, label='BOSZ model')
        if len(self.point_estimates.ppxf_fit) == 2:      # check if there's a MARCS solution
            axes.plot(self.clean_spec.wave, self.point_estimates.ppxf_fit['MARCS'], c='r', lw=2,
                      label='MARCS model')
        axes.legend(loc=0, fontsize=20)
        axes.set_ylabel('Relative flux', fontsize=25)
        axes.tick_params(axis='both', which='major', labelsize=20, direction='in')

        ax2.plot(self.clean_spec.wave, self.clean_spec.corrected_flux_med - self.point_estimates.ppxf_fit['BOSZ'],
                 c='b', lw=2)
        if len(self.point_estimates.ppxf_fit) == 2:      # check if there's a MARCS solution
            ax2.plot(self.clean_spec.wave, self.clean_spec.corrected_flux_med - self.point_estimates.ppxf_fit['MARCS']
                     , c='r', lw=2)
        ax2.set_ylabel('Residual flux', fontsize=25)
        ax2.set_xlabel(r'$Wavelength, \AA$', fontsize=25)
        ax2.tick_params(axis='both', which='major', labelsize=20, direction='in')
        ax2.axhline(y=0, c='r', linestyle='--', lw=2)
        axes.set_xticks([])
        plt.tight_layout()
        plt.savefig(var.plots_output_direc + str(self.pim) + '/bestfit.png', bbox_inches='tight')
        plt.close()

    def corner(self):
        """Plot corner."""
        plt.clf()
        if var.alpha:
            fig = corner(self.point_estimates.samples_clean, labels=["$T_{eff}$", "$log g$", "$[Fe/H]$", "[alpha/M]"],
                         quantiles=None, show_titles=False, title_kwargs={"fontsize": 16},
                         label_kwargs={"fontsize": 18}, color='b')
        else:
            fig = corner(self.point_estimates.samples_clean, labels=["$T_{eff}$", "$log g$", "$[Fe/H]$"],
                         quantiles=None, show_titles=False, title_kwargs={"fontsize": 16},
                         label_kwargs={"fontsize": 18}, color='b')
        axes = np.array(fig.axes).reshape((var.ndim, var.ndim))   # Extract the axes
        for i in range(var.ndim):       # Loop over the diagonal
            ax = axes[i, i]
            if i == 0:
                ax.set_title('$T_{eff} = %d^{+%d}_{-%d}$K' % (
                    np.round(self.point_estimates.params[self.model][i][0], 0),
                    np.round(self.point_estimates.params[self.model][i][2], 0),
                    np.round(self.point_estimates.params[self.model][i][1], 0)), fontsize=14)
            if i == 1:
                ax.set_title('$Log g = %6.2f^{+%6.2f}_{-%6.2f}$K' % (
                    np.round(self.point_estimates.params[self.model][i][0], 2),
                    np.round(self.point_estimates.params[self.model][i][2], 2),
                    np.round(self.point_estimates.params[self.model][i][1], 2)), fontsize=14)
            if i == 2:
                ax.set_title('$[Fe/H] = %6.2f^{+%6.2f}_{-%6.2f}$K' % (
                    np.round(self.point_estimates.params[self.model][i][0], 2),
                    np.round(self.point_estimates.params[self.model][i][2], 2),
                    np.round(self.point_estimates.params[self.model][i][1], 2)), fontsize=14)
            if i == 3:
                ax.set_title('$[alpha/M] = %6.2f^{+%6.2f}_{-%6.2f}$K' % (
                    np.round(self.point_estimates.params[self.model][i][0], 2),
                    np.round(self.point_estimates.params[self.model][i][2], 2),
                    np.round(self.point_estimates.params[self.model][i][1], 2)), fontsize=14)
            ax.axvline(self.point_estimates.params[self.model][i][0], color="r")
            ax.axvline(self.point_estimates.params[self.model][i][0] - self.point_estimates.params[self.model][i][1],
                       color="k", linestyle='--')
            ax.axvline(self.point_estimates.params[self.model][i][0] + self.point_estimates.params[self.model][i][2],
                       color="k", linestyle='--')

        for yi in range(var.ndim):      # Loop over the histograms
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(self.point_estimates.params[self.model][xi][0], color="r")
                ax.axhline(self.point_estimates.params[self.model][yi][0], color="r")
                ax.plot(self.point_estimates.params[self.model][xi][0], self.point_estimates.params[self.model][yi][0],
                        "sr")

        plt.tight_layout()
        plt.savefig(var.plots_output_direc + str(self.pim) + '/' + self.model + '_corner.png', bbox_inches='tight')
        plt.close()

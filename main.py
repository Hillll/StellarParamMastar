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
import datetime, random, sys, time
from scipy.stats import chisquare as sp_chi

from functions import load_data, prepare_spectrum, mcmc_v2
from spd_setup import spd_setup

from math import pi
from scipy.constants import c

import time
t0 = time.time()

target_num = int(sys.argv[1])   # used to select array of spectra to analyse
t0 = time.time()

# instantiate params
var = spd_setup()
'''
if not var.alpha:
    marcs_mods = synth_models('marcs')
    bosz_mods = synth_models('bosz')'''

# instantiate params
var = spd_setup()

# get mastar data and targets to analyse
mast_data = load_data()
targets = mast_data.get_targets(number=target_num)
mast_data.get_mastar(start=targets[0], end=targets[-1])

# get estimates data
mast_data.get_estimates(start=targets[0], end=targets[-1])
ebv_gaia = mast_data.meta_data['ebv']

ebv_gaia = np.array([0.02, -99, 0.02, 0.02, 0.02])
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

    temp = mcmc_v2(model='BOSZ', flux=clean_spec.corrected_flux_med, yerr=clean_spec.yerr,
                     meta_data=mast_data.meta_data[c], parallel=True)
    temp.starting()
    sampler = temp.sample()
'''
    # run mcmc
    mcmc_temp = mcmc(model='BOSZ', flux=clean_spec.corrected_flux_med, yerr=clean_spec.yerr,
                     meta_data=mast_data.meta_data[c], marcs_mods=marcs_mods, bosz_mods=bosz_mods)
    mcmc_temp.starting()    # generate starting values for each set of walkers
    sampler = mcmc_temp.sample()
'''
print('\nTotal time taken: ', time.time() - t0)





'''
    # get point estimates from chains
    point = point_estimates(sampler)
    point.params_err(teff='median', logg='mode', zh='mode', alpha='mode')
    point.get_chi2()
    if var.plot:
        point.plotting()'''
'''
### Define functions
def model_marcs(theta):  # model given a set of parameters (theta)
    t, g, z = theta
    # ap = (np.log10(t), g, 0, (10**(z))*0.0169)
    ap = (np.log10(t), g, 0, (10 ** (z)) * 0.02)  # try different normalisation
    temp_flux = marcs_mods.generate_stellar_spectrum(*ap)
    temp_wave = np.asarray(marcs_mods.wavelength)

    temp_flux_med = temp_flux / np.median(temp_flux)

    return temp_wave, temp_flux_med


def model_bosz(theta):  # model given a set of parameters (theta)
    t, g, z = theta
    # ap = (np.log10(t), g, 0, (10**(z))*0.0169)
    ap = (np.log10(t), g, 0, (10 ** (z)) * 0.02)  # try different normalisation
    temp_flux = bosz_mods.generate_stellar_spectrum(*ap)
    temp_wave = np.asarray(bosz_mods.wavelength)

    temp_flux_med = temp_flux / np.median(temp_flux)

    return temp_wave, temp_flux_med


def lnlike_marcs(theta, y, yerr):  # likelihood fn that evaluates best fit
    t, g, z = theta

    model_flux = np.asarray(model_marcs(theta)[1])  # model flux with mcmc proposed parameters (theta)
    model_wave = np.asarray(model_marcs(theta)[0])

    if np.array_equal(model_flux,
                      np.ones(len(model_flux))):  # if theta is outside model grid bbox, array of ones is returned
        return -np.inf  # return -inf so walkers move away form this parameter combination

    else:
        start = [0, 10]
        sol = (ppxf(model_flux, y, noise=yerr, velscale=velscale, start=start, degree=-1, mdegree=6, moments=2,
                    quiet=True))  # run ppxf with interpolated model

        t = ((y - sol.bestfit) ** 2) / (yerr ** 2)
        mask = ~np.isinf(t) & ~np.isnan(t)  # remove inf values from array
        LnLike = -0.5 * np.sum(np.log(2 * pi * (yerr ** 2)) + t[mask]) / len(yerr)
        return LnLike


def lnlike_bosz(theta, y, yerr):  # likelihood fn that evaluates best fit
    t, g, z = theta

    model_flux = np.asarray(model_bosz(theta)[1])  # model flux with mcmc proposed parameters (theta)
    model_wave = np.asarray(model_bosz(theta)[0])

    if np.array_equal(model_flux,
                      np.ones(len(model_flux))):  # if theta is outside model grid bbox, array of ones is returned
        return -np.inf  # return -inf so walkers move away form this parameter combination

    else:
        start = [0, 10]
        sol = (ppxf(model_flux, y, noise=yerr, velscale=velscale, start=start, degree=-1, mdegree=6, moments=2,
                    quiet=True))  # run ppxf with interpolated model

        t = ((y - sol.bestfit) ** 2) / (yerr ** 2)
        mask = ~np.isinf(t) & ~np.isnan(t)  # remove inf values from array
        LnLike = -0.5 * np.sum(np.log(2 * pi * (yerr ** 2)) + t[mask]) / len(yerr)
        return LnLike


def lnprior_marcs(theta, i):  # prior estimate of the data - flat
    t, g, z = theta
    if teff_est_min_i <= t <= teff_est_max_i and logg_est_min_i <= g <= logg_est_max_i and t > 2000 and -3 <= z <= 1:
        return 1
    else:
        return -np.inf


def lnprior_bosz(theta, i):  # prior estimate of the data - flat
    t, g, z = theta
    if teff_est_min_i <= t <= teff_est_max_i and logg_est_min_i <= g <= logg_est_max_i and t > 3500 and -3 <= z <= 0.5:
        return 1
    else:
        return -np.inf


def lnprob_marcs(theta, y, yerr):  # posterior probability marcs
    t, g, z = theta
    lp = lnprior_marcs(theta, i)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_marcs(theta, y, yerr)


def lnprob_bosz(theta, y, yerr):  # posterior probability bosz
    t, g, z = theta
    lp = lnprior_bosz(theta, i)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_bosz(theta, y, yerr)


def chi2_lh(data, model, error):
    chi = np.sum(((data - model) ** 2 / error ** 2))
    return chi


def max_post(samp):
    X = np.linspace(min(samp), max(samp))
    func = scipy.stats.gaussian_kde(samp)
    return X[np.argmax(func(X))]






targts_all = np.array([[5513, 50143], [17918, 31840], [805, 54388]])
t0 = time.time()  # time codemastar-goodspec-mpl11-gaia-DT-v2-mpl11-v3_1_1-v1_7_5.fits
targts = targts_all[number]
targts = [0, 1]

for i in targts:
    starid = i
    print(i)

    # loc = np.where((plate == plate_slct[i])&(ifu == ifu_slct[i])&(mjd == mjd_slct[i]))[0][0]  #find location of spectrum in DT file
    loc = i
    initial = np.array([teff_est[loc], logg_est[loc], met_est[loc]])
    print(initial)

    if teff_est_min[loc] > teff_est[loc] * 0.84:
        teff_est_min_i = teff_est[loc] * 0.84
    else:
        teff_est_min_i = teff_est_min[loc]

    if teff_est_max[loc] < teff_est[loc] * 1.16:
        teff_est_max_i = teff_est[loc] * 1.16
    else:
        teff_est_max_i = teff_est_max[loc]

    logg_est_min_i, logg_est_max_i = logg_est_min[loc], logg_est_max[loc]
    met_est_min_i, met_est_max_i = met_est_min[loc], met_est_max[loc]

    # print('teff ests: ', teff_est_min_i, teff_est_max_i)

    if teff_est_min_i < 2500:  # restrict prior range to models +/- some leeway
        teff_est_min_i = 2000
    if teff_est_max_i > 35000:
        teff_est_max_i = 40000
    if logg_est_min_i < -0.5:
        logg_est_min_i = -1
    if logg_est_max_i > 5.5:
        logg_est_max_i = 6

    ebv = ebv_gaia[loc]  # correct for dust using gaia

    if initial[0] < 0:  # deal with no gaia info
        print('No Gaia information, ', i)
        np.savez('/mnt/lustre/lhill/shera/mcmc/bin/mast_' + str(i), np.array([-999, -999, -999]))
        continue

    if (initial[1] < -10) or (np.isnan(initial[1])):  # deal with logg est not existing or nan
        initial[1] = 3.5

    ##try FFly dust
    red_fm = reddening_fm(wave_mast, ebv=ebv)

    flux_slct = flux_mast[i, :][9:-8]
    corrected_flux = flux_slct * red_fm

    red_fm = reddening_fm(wave_mast, ebv=ebv)

    flux_slct = flux_mast[i, :][9:-8]
    corrected_flux = flux_slct * red_fm

    if np.all(flux_slct > 0) == False:  # deal with dead pixels
        flux_x = np.arange(len(flux_slct))
        idxx = np.where(flux_slct > 0)
        ff = interp1d(flux_x[idxx], flux_slct[idxx], fill_value='extrapolate')  # interp function with non zero values
        flux_slct_new = ff(flux_x)  # interpolate where zero values occur

        corrected_flux = flux_slct_new * red_fm

        ivar_ = ivar[i][9:-8]  # get inverse variance
        ivar_x = np.arange(len(ivar_))
        idx = np.nonzero(ivar_)  # nonzero values in ivar
        f = interp1d(ivar_x[idx], ivar_[idx], fill_value='extrapolate')  # interp function with non zero values
        ivar_new = f(ivar_x)  # interpolate where zero values occur
        sd = ivar_new ** -0.5  # change inverse variance to standard error
        sd_pcnt = sd / flux_slct_new  # error as a percentage of the flux

        corrected_flux_med = corrected_flux / np.median(corrected_flux)  # median normalise
        yerr = corrected_flux_med * sd_pcnt
    else:
        ivar_ = ivar[i][9:-8]  # get inverse variance
        ivar_x = np.arange(len(ivar_))
        idx = np.nonzero(ivar_)  # nonzero values in ivar
        f = interp1d(ivar_x[idx], ivar_[idx], fill_value='extrapolate')  # interp function with non zero values
        ivar_new = f(ivar_x)  # interpolate where zero values occur
        sd = ivar_new ** -0.5  # change inverse variance to standard error
        sd_pcnt = sd / flux_slct  # error as a percentage of the flux

        corrected_flux_med = corrected_flux / np.median(corrected_flux)  # median normalise
        yerr = corrected_flux_med * sd_pcnt

    start = [0, 10]
    velscale = 69

    # catch any values that didnt get corrected and skip, spectrum must be bad
    if np.any(np.isnan(sd_pcnt)):
        np.savez('/mnt/lustre/lhill/shera/mcmc/bin/mast_' + str(i), np.array([-999, -999, -999]))
        continue

    velscale = 69

    if initial[0] < -10 or initial[1] < -10 or initial[2] < -10:  # deal with n/a priors
        print('Invalid prior estimates. Moving to next object... \n')
        np.savez('/mnt/lustre/lhill/shera/mcmc/bin/mast_' + str(i), np.array([-999, -999, -999]))
        continue

    if initial[0] > 40000 or initial[0] < 2000:  # deal with data outside of template range
        print('Estimate for temperature outside of template range. Moving to next object... \n')
        np.savez('/mnt/lustre/lhill/shera/mcmc/bin/mast_' + str(i), np.array([-999, -999, -999]))
        continue

    p0_ = []  # generate random starting points for the walkers, uniform across min and max priors
    for j in range(var.nwalkers):
        temp = [random.uniform(teff_est_min_i, teff_est_max_i), random.uniform(logg_est_min_i, logg_est_max_i), \
                random.uniform(-2.5, 0.5)]
        p0_.append(temp)
    p0_ = np.asarray(p0_)


    def main_marcs(p0_, nwalkers, niter, ndim, lnprob_marcs, data):  # The MCMC routine
        with Pool() as pool:
            sampler = emcee_LH.EnsembleSampler(nwalkers, ndim, lnprob_marcs, args=data, a=var.a, pool=pool)

            # Burn in
            p0_, _, _ = sampler.run_mcmc(p0_, var.burnin)  # this diminishes the influence of starting values

            # sampler.reset()                  #reset sampler to save memory

            # Production
            pos, prob, state = sampler.run_mcmc(p0_, niter)
            # print('Finished production! \n')
            return sampler, pos, prob, state


    def main_bosz(p0_, nwalkers, niter, ndim, lnprob_bosz, data):  # The MCMC routine
        with Pool() as pool:
            sampler = emcee_LH.EnsembleSampler(nwalkers, ndim, lnprob_bosz, args=data, a=var.a, pool=pool)

            # Burn in
            p0_, _, _ = sampler.run_mcmc(p0_, var.burnin)  # this diminishes the influence of starting values

            # sampler.reset()                  #reset sampler to save memory

            # Production
            pos, prob, state = sampler.run_mcmc(p0_, niter)
            # print('Finished production! \n')
            return sampler, pos, prob, state


    ###Run routine...
    print('running mcmc for Bosz models')
    sampler, pos, prob, state = main_bosz(p0_, var.nwalkers, var.niter, var.ndim, lnprob_bosz, (corrected_flux_med, yerr))

    samples_all = sampler.flatchain

    samples_arr_bosz = sampler.chain
    samples_cut = samples_arr_bosz[:, var.burnin:, :]  # remove burn in
    samples_cut_flat = samples_cut.reshape((-1, 3))  # flatten array

    bosz_params_err = []
    for k in range(var.ndim):
        mcmc = np.percentile(samples_cut_flat[:, k], [16, 50, 84])
        q = np.diff(mcmc)
        bosz_params_err.append([mcmc[1], q[0], q[1]])
        # print(mcmc[1], q[0], q[1])

    # calculate met
    met = max_post(samples_cut_flat[:, 2])

    # calculate interval errors
    met_err_dn = (met - pymc3.stats.hpd(samples_cut_flat[:, 2], 0.32)[0])
    met_err_up = (pymc3.stats.hpd(samples_cut_flat[:, 2], 0.32)[1] - met)
    bosz_params_err.append([met, met_err_dn, met_err_up])
    bosz_params_err = np.asarray(bosz_params_err)

    value2 = [np.median(samples_cut_flat[:, 0]), np.median(samples_cut_flat[:, 1]), met]

    # calculate chisquare
    model_flux_bosz = np.asarray(model_bosz(value2)[1])
    model_wave = np.asarray(model_bosz(value2)[0])
    start = [0, 10]
    ones = np.ones_like(model_flux_bosz)
    sol = (ppxf(model_flux_bosz, corrected_flux_med, noise=yerr, velscale=velscale, start=start, degree=-1, mdegree=6,
                moments=2, quiet=True))
    # sol = (ppxf(model_flux,corrected_flux_med,noise=yerr,velscale=velscale,start=start,degree=-1,mdegree=10,moments=2,quiet=True))   #multiplicative
    ppxf_fit_bosz = sol.bestfit
    ppxf_chi = sol.chi2
    # scipy_chi, scipy_p = sp_chi(corrected_flux_med, sol.bestfit)
    # chis = np.array([ppxf_chi, scipy_chi, scipy_p])

    chis_bosz = np.array([ppxf_chi, chi2_lh(corrected_flux_med, sol.bestfit, yerr) / (len(corrected_flux_med - 3))])

    bosz_params = value2

    folder = '/mnt/lustre/lhill/shera/mcmc/plots/bin/mast_' + str(starid)  # check and create folder for data

    if not os.path.exists(folder):
        os.makedirs(folder)

    ###trace plot
    plt.clf()
    plt.figure(figsize=(16, 30))
    plt.subplot(3, 1, 1)
    plt.plot(sampler.chain[:, :, 0].T, '--', color='k', alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.ylabel('Effective Temperature (Kelvin)', fontsize=16)
    plt.xlabel('Iterations', fontsize=16)

    plt.subplot(3, 1, 2)
    plt.plot(sampler.chain[:, :, 1].T, '--', color='k', alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.ylabel('Log g', fontsize=16)
    plt.xlabel('Iterations', fontsize=16)

    plt.subplot(3, 1, 3)
    plt.plot(sampler.chain[:, :, 2].T, '--', color='k', alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.ylabel('metallicity (Z)', fontsize=16)
    plt.xlabel('Iterations', fontsize=16)
    # plt.savefig(folder+'/trace_bosz.png', bbox_inches='tight')

    ###corner plot
    plt.clf()
    fig = corner(samples_cut_flat, labels=["$T_{eff}$", "$log g$", "$[Fe/H]$"], quantiles=None, \
                 show_titles=False, title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 18}, color='b')
    axes = np.array(fig.axes).reshape((var.ndim, var.ndim))
    for i in range(var.ndim):
        ax = axes[i, i]
        if i == 0:
            ax.set_title('$T_{eff} = %d^{+%d}_{-%d}$K' % (
                np.round(bosz_params_err[i, 0], 0), np.round(bosz_params_err[i, 2], 0),
                np.round(bosz_params_err[i, 1], 0)),
                         fontsize=14)
        if i == 1:
            ax.set_title('$Log g = %6.2f^{+%6.2f}_{-%6.2f}$K' % (
                np.round(bosz_params_err[i, 0], 2), np.round(bosz_params_err[i, 2], 2),
                np.round(bosz_params_err[i, 1], 2)),
                         fontsize=14)
        if i == 2:
            i = 3
            ax.set_title('$[\it{Z}/H] = %6.2f^{+%6.2f}_{-%6.2f}$K' % (
                np.round(bosz_params_err[i, 0], 2), np.round(bosz_params_err[i, 2], 2),
                np.round(bosz_params_err[i, 1], 2)),
                         fontsize=14)
        # ax.set_title('Test', fontsize=16)
        ax.axvline(bosz_params_err[i, 0], color="r")
        ax.axvline(bosz_params_err[i, 0] - bosz_params_err[i, 1], color="k", linestyle='--')
        ax.axvline(bosz_params_err[i, 0] + bosz_params_err[i, 2], color="k", linestyle='--')

        # Loop over the histograms
    for yi in range(var.ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value2[xi], color="r")
            ax.axhline(value2[yi], color="r")
            ax.plot(value2[xi], value2[yi], "sr")
        # plt.savefig(folder+'/corner_bosz.png')

    if teff_est_min_i > 5000:  # skip marcs run if star is hot

        plt.figure(figsize=(22, 9))
        plt.plot((np.asarray(model_bosz(bosz_params)[0])), corrected_flux_med, linewidth=1.5, c='k',
                 label='MaStar spectrum')
        plt.plot((np.asarray(model_bosz(bosz_params)[0])), ppxf_fit_bosz, linewidth=1.5, c='b', linestyle='-',
                 label='Best fit BOSZ')
        plt.tick_params(axis='both', which='major', labelsize=26)
        plt.xlabel(r'$Wavelength, \AA$', fontsize=25)
        plt.ylabel(r'$Relative flux$', fontsize=25)
        plt.legend(fontsize=20)
        plt.tight_layout()
        # plt.savefig(folder+'/bestfit_bosz.png')
        # np.savez('/mnt/lustre/lhill/shera/mcmc/bin/mast_'+str(starid), np.array([-999]), bosz_params_err, np.array([-999]), chis_bosz, np.array([-999]), samples_arr_bosz)
        np.savez('/mnt/lustre/lhill/shera/mcmc/bin/mast_' + str(starid) + 'params', bosz_params_err, chis_bosz)
        np.savez('/mnt/lustre/lhill/shera/mcmc/bin/mast_' + str(starid) + 'chains', samples_arr_bosz)
        np.savez('/mnt/lustre/lhill/shera/mcmc/bin/mast_' + str(starid) + 'fits',
                 np.asarray(model_bosz(bosz_params)[0]), corrected_flux_med, ppxf_fit_bosz)
        continue

    # marcs

    print('running mcmc for Marcs models')
    sampler, pos, prob, state = main_marcs(p0_, var.nwalkers, var.niter, var.ndim, lnprob_marcs, (corrected_flux_med, yerr))

    samples_all = sampler.flatchain

    samples_arr_marcs = sampler.chain
    samples_cut = samples_arr_marcs[:, var.burnin:, :]  # remove burn in
    samples_cut_flat = samples_cut.reshape((-1, 3))  # flatten array

    marcs_params_err = []
    for k in range(var.ndim):
        mcmc = np.percentile(samples_cut_flat[:, k], [16, 50, 84])
        q = np.diff(mcmc)
        marcs_params_err.append([mcmc[1], q[0], q[1]])
        # print(mcmc[1], q[0], q[1])

    # calculate met
    met = max_post(samples_cut_flat[:, 2])

    # calculate interval errors
    met_err_dn = (met - pymc3.stats.hpd(samples_cut_flat[:, 2], 0.32)[0])
    met_err_up = (pymc3.stats.hpd(samples_cut_flat[:, 2], 0.32)[1] - met)
    marcs_params_err.append([met, met_err_dn, met_err_up])
    marcs_params_err = np.asarray(marcs_params_err)

    value1 = [np.median(samples_cut_flat[:, 0]), np.median(samples_cut_flat[:, 1]), met]

    # calculate chisquare
    model_flux_marcs = np.asarray(model_marcs(value1)[1])
    model_wave = np.asarray(model_marcs(value1)[0])
    # yerr = np.ones_like(model_flux)
    start = [0, 10]
    ones = np.ones_like(model_flux_marcs)
    sol = (ppxf(model_flux_marcs, corrected_flux_med, noise=yerr, velscale=velscale, start=start, degree=-1, mdegree=6,
                moments=2, quiet=True))
    # sol = (ppxf(model_flux,corrected_flux_med,noise=yerr,velscale=velscale,start=start,degree=-1,mdegree=10,moments=2,quiet=True))   #multiplicative
    ppxf_fit = sol.bestfit
    ppxf_chi = sol.chi2
    # scipy_chi, scipy_p = sp_chi(corrected_flux_med, sol.bestfit)
    # chis = np.array([ppxf_chi, scipy_chi, scipy_p])

    chis_marcs = np.array([ppxf_chi, chi2_lh(corrected_flux_med, sol.bestfit, yerr) / (len(corrected_flux_med - 3))])

    marcs_params = value1

    ###trace plot
    plt.clf()
    plt.figure(figsize=(16, 30))
    plt.subplot(3, 1, 1)
    plt.plot(sampler.chain[:, :, 0].T, '--', color='k', alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.ylabel('Effective Temperature (Kelvin)', fontsize=16)
    plt.xlabel('Iterations', fontsize=16)

    plt.subplot(3, 1, 2)
    plt.plot(sampler.chain[:, :, 1].T, '--', color='k', alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.ylabel('Log g', fontsize=16)
    plt.xlabel('Iterations', fontsize=16)

    plt.subplot(3, 1, 3)
    plt.plot(sampler.chain[:, :, 2].T, '--', color='k', alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.ylabel('metallicity (Z)', fontsize=16)
    plt.xlabel('Iterations', fontsize=16)
    # plt.savefig(folder+'/trace_marcs.png', bbox_inches='tight')

    ###corner plot
    plt.clf()
    fig = corner(samples_cut_flat, labels=["$T_{eff}$", "$log g$", "$[Fe/H]$"], quantiles=None, \
                 show_titles=False, title_kwargs={"fontsize": 16}, label_kwargs={"fontsize": 18}, color='b')
    axes = np.array(fig.axes).reshape((var.ndim, var.ndim))
    for i in range(var.ndim):
        ax = axes[i, i]
        if i == 0:
            ax.set_title('$T_{eff} = %d^{+%d}_{-%d}$K' % (
                np.round(marcs_params_err[i, 0], 0), np.round(marcs_params_err[i, 2], 0),
                np.round(marcs_params_err[i, 1], 0)), fontsize=14)
        if i == 1:
            ax.set_title('$Log g = %6.2f^{+%6.2f}_{-%6.2f}$K' % (
                np.round(marcs_params_err[i, 0], 2), np.round(marcs_params_err[i, 2], 2),
                np.round(marcs_params_err[i, 1], 2)), fontsize=14)
        if i == 2:
            i = 3
            ax.set_title('$[\it{Z}/H] = %6.2f^{+%6.2f}_{-%6.2f}$K' % (
                np.round(marcs_params_err[i, 0], 2), np.round(marcs_params_err[i, 2], 2),
                np.round(marcs_params_err[i, 1], 2)), fontsize=14)
        # ax.set_title('Test', fontsize=16)
        ax.axvline(marcs_params_err[i, 0], color="r")
        ax.axvline(marcs_params_err[i, 0] - marcs_params_err[i, 1], color="k", linestyle='--')
        ax.axvline(marcs_params_err[i, 0] + marcs_params_err[i, 2], color="k", linestyle='--')

        # Loop over the histograms
    for yi in range(var.ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value1[xi], color="r")
            ax.axhline(value1[yi], color="r")
            ax.plot(value1[xi], value1[yi], "sr")
        # plt.savefig(folder+'/corner_marcs.png')

    ###best fit plot with polynomial adjusted spectrum

    model_flux_marcs = np.asarray(model_marcs(marcs_params)[1])
    noise = yerr
    start = [0, 10]
    sol = (ppxf(model_flux_marcs, corrected_flux_med, noise=yerr, velscale=velscale, start=start, degree=-1, mdegree=6,
                moments=2, quiet=True))

    ppxf_fit_marcs = sol.bestfit

    np.savez('/mnt/lustre/lhill/shera/mcmc/bin/mast_' + str(starid) + 'params', marcs_params_err, bosz_params_err,
             chis_marcs, chis_bosz)
    np.savez('/mnt/lustre/lhill/shera/mcmc/bin/mast_' + str(starid) + 'chains', samples_arr_marcs, samples_arr_bosz)
    np.savez('/mnt/lustre/lhill/shera/mcmc/bin/mast_' + str(starid) + 'fits', np.asarray(model_bosz(bosz_params)[0]),
             corrected_flux_med, ppxf_fit_marcs, ppxf_fit_bosz)

    ###best fit plot with polynomial adjusted spectrum
    print('BOSZ  params = ', bosz_params)
    print('MARCS params = ', marcs_params)

    plt.figure(figsize=(22, 9))
    plt.plot((np.asarray(model_bosz(bosz_params)[0])), corrected_flux_med, linewidth=1.5, c='k',
             label='MaStar spectrum')
    plt.plot((np.asarray(model_marcs(marcs_params)[0])), ppxf_fit_marcs, linewidth=1.5, c='r', linestyle='-',
             label='Best fit MARCS')
    plt.plot((np.asarray(model_bosz(bosz_params)[0])), ppxf_fit_bosz, linewidth=1.5, c='b', linestyle='-',
             label='Best fit BOSZ')
    plt.tick_params(axis='both', which='major', labelsize=26)
    plt.xlabel(r'$Wavelength, \AA$', fontsize=25)
    plt.ylabel(r'$Relative flux$', fontsize=25)
    plt.legend(fontsize=20)
    plt.tight_layout()
    # plt.savefig(folder+'/bestfit_both.png')

    print('Done. Time taken: ', np.round(time.time() - t0, 3))
'''
import numpy as np
from ppxf.ppxf import ppxf
import arviz, scipy
from spd_setup import spd_setup
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from interp import interp_models
from mpl_toolkits.axes_grid1 import make_axes_locatable
from corner import corner

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

var = spd_setup()
if var.alpha:
    print('\nGetting models...')
    marcs_m04 = interp_models('marcs_m04')
    marcs_p04 = interp_models('marcs_p04')

    bosz_m03 = interp_models('bosz_m03')
    bosz_p03 = interp_models('bosz_p03')
    bosz_p05 = interp_models('bosz_p05')

marcs_p0 = interp_models('marcs_p0')
bosz_p0 = interp_models('bosz_p0')

def model_spec(theta, model):  # model given a set of parameters (theta)
    if var.alpha:
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
        t, g, z, a = theta
        ap = (t, g, z)
        if model == 'bosz' or model == 'BOSZ':
            flux = bosz_p0.generate_stellar_spectrum(ap)
        elif model == 'marcs' or model == 'MARCS':
            flux = marcs_p0.generate_stellar_spectrum(ap)
        else:
            raise Exception("Invalid model library.")
        flux_med = flux / np.median(flux)
        return flux_med


class load_data:
    """Load spectra and the estimates file from Gaia photometry."""

    def __init__(self, pims_todo, data_direc=var.data_direc, spec_file=var.spec_file, est_file=var.est_file):
        self.data_direc = data_direc
        self.spec_file = spec_file
        self.est_file = est_file
        self.pims_todo = pims_todo

    def get_mastar(self):
        header = fits.open(self.data_direc + self.spec_file)
        self.mangaid = header[1].data['mangaid']
        self.plate_1, self.ifu_1, self.mjd_1 = header[1].data['plate'], header[1].data['ifudesign'], header[1].data[
            'mjd']
        self.ifu_1 = np.asarray([int(i) for i in self.ifu_1])  # ensure ifu is int
        self.pim_all = np.array([int(str((self.plate_1[i])) + str((self.ifu_1[i])) + str((self.mjd_1[i]))) for i in
                                 range(len(self.plate_1))])
        self.target_mask = self.get_targets()
        self.ra, self.dec = header[1].data['ra'][self.target_mask], header[1].data['dec'][self.target_mask]
        self.wave = header[1].data['wave'][0][9:-8]

        global lambda_mask
        if var.min_lambda == -999 and var.max_lambda == -999:
            lambda_mask = np.where(self.wave > var.max_lambda)[0]  # remove start and end pixels as downgrading the
            # models gives bad pixels here
        elif var.min_lambda == -999 and var.max_lambda != -999:
            lambda_mask = np.where(self.wave <= var.max_lambda)[0]
        elif var.min_lambda != -999 and var.max_lambda == -999:
            lambda_mask = np.where(var.min_lambda >= self.wave)[0]
        else:
            lambda_mask = np.where((self.wave >= var.min_lambda) & (self.wave <= var.max_lambda))

        self.flux = header[1].data['flux'][self.target_mask]
        self.flux = np.asarray([i[lambda_mask] for i in self.flux])
        self.wave = self.wave[lambda_mask]
        self.ivar, self.exptime = header[1].data['ivar'][self.target_mask], header[1].data['exptime'][self.target_mask]
        self.ivar = np.asarray([i[lambda_mask] for i in self.ivar])
        header.close()

        self.pim = self.pim_all[self.target_mask]


    def get_estimates(self, start=0, end=1000):
        # if not isinstance(plate, array)
        header = fits.open(self.data_direc + self.est_file)
        pim_mask = [np.where(header[1].data['pim'] == i)[0][0] for i in self.pim]
        self.meta_data = header[1].data[pim_mask]
        header.close()

    def get_ebv_allspec(self):
        header = fits.open('/home/lewishill/Downloads/mastarall_v1-7-7-ALLVISITS-PIM.fits')
        ebv = header[1].data['ebv']
        self.pim_allspec = header[1].data['pim']


    def get_targets(self):
        return [np.where(self.pim_all == i)[0][0] for i in self.pims_todo]

class load_good_bad:
    """ Load good and bad spec data and dered"""

    def __init__(self, pims_todo, data_direc=var.data_direc, spec_file=var.spec_file, est_file=var.est_file):
        self.data_direc = data_direc
        self.spec_file = spec_file
        self.est_file = est_file
        self.pims_todo = pims_todo

    def load_all(self):
        print('\nLoading bad spec data...')
        header_bad = fits.open('/home/lewishill/Downloads/mastar-badspec-v3_1_1-v1_7_7.fits')
        self.mangaid_bad = header_bad[1].data['mangaid']
        self.plate_1_bad, self.ifu_1_bad, self.mjd_1_bad = header_bad[1].data['plate'], header_bad[1].data['ifudesign'], \
                                                           header_bad[1].data['mjd']
        self.ifu_1_bad = np.asarray([int(i) for i in self.ifu_1_bad])  # ensure ifu is int
        self.pim_all_bad = np.array(
            [int(str((self.plate_1_bad[i])) + str((self.ifu_1_bad[i])) + str((self.mjd_1_bad[i]))) for i in
             range(len(self.plate_1_bad))])
        self.target_mask_bad = self.get_targets('bad')
        self.mangaid_bad_slct = self.mangaid_bad[self.target_mask_bad]
        self.flux_bad = header_bad[1].data['flux'][self.target_mask_bad]
        header_bad.close()

        print('\nLoading good spec data...')
        header_good = fits.open(self.data_direc + self.spec_file)
        self.mangaid_good = header_good[1].data['mangaid']
        self.plate_1_good, self.ifu_1_good, self.mjd_1_good = header_good[1].data['plate'], header_good[1].data['ifudesign'], header_good[1].data[
            'mjd']
        self.ifu_1_good = np.asarray([int(i) for i in self.ifu_1_good])  # ensure ifu is int
        self.pim_all_good = np.array([int(str((self.plate_1_good[i])) + str((self.ifu_1_good[i])) + str((self.mjd_1_good[i]))) for i in
                                 range(len(self.plate_1_good))])
        self.target_mask_good = self.get_targets('good')
        self.wave = header_good[1].data['wave'][0]
        self.mangaid_good_slct = self.mangaid_good[self.target_mask_good]
        self.flux_good = header_good[1].data['flux'][self.target_mask_good]
        header_good.close()
        print('\nLoaded data.')

    def plot_dered(self):
        header = fits.open('/home/lewishill/Downloads/mastarall_v1-7-7-ALLVISITS-PIM.fits')
        ebv_all = header[1].data['ebv']
        pim_allspec = header[1].data['pim']
        ebv_good = [ebv_all[np.where(pim_allspec == i)] for i in self.pim_all_good[self.target_mask_good]]
        ebv_bad = [ebv_all[np.where(pim_allspec == i)] for i in self.pim_all_bad[self.target_mask_bad]]

        print('\nPlotting spectra...')
        if len(self.target_mask_good) > 0:
            print('\nPlotting {} spectra from good visits'.format(len(self.flux_good)))
            for c,i in enumerate(self.flux_good):
                plt.figure(figsize=(18,8))
                plt.plot(self.wave, self.dered(i, ebv_good[c]))
                plt.ylabel('Flux', fontsize=25)
                plt.xlabel(r'$Wavelength, \AA$', fontsize=25)
                plt.tick_params(axis='both', which='major', labelsize=20, direction='in')
                plt.tight_layout()
                plt.savefig('plots/ostars/'+str(self.pim_all_good[self.target_mask_good][c])+'.png', bbox_inches='tight')

        if len(self.target_mask_bad) > 0:
            print('Plotting {} spectra from bad visits'.format(len(self.flux_bad)))
            for c,i in enumerate(self.flux_bad):
                plt.figure(figsize=(18,8))
                plt.plot(self.wave, self.dered(i, ebv_bad[c]))
                plt.ylabel('Flux', fontsize=25)
                plt.xlabel(r'$Wavelength, \AA$', fontsize=25)
                plt.tick_params(axis='both', which='major', labelsize=20, direction='in')
                plt.tight_layout()
                plt.savefig('plots/ostars/'+str(self.pim_all_bad[self.target_mask_bad][c])+'.png', bbox_inches='tight')

    def plot_grid(self):
        header = fits.open('/home/lewishill/Downloads/mastarall_v1-7-7-ALLVISITS-PIM.fits')
        ebv_all = header[1].data['ebv']
        pim_allspec = header[1].data['pim']
        ebv_good = [ebv_all[np.where(pim_allspec == i)] for i in self.pim_all_good[self.target_mask_good]]
        ebv_bad = [ebv_all[np.where(pim_allspec == i)] for i in self.pim_all_bad[self.target_mask_bad]]
        pim_ordered = np.concatenate((self.pim_all_good[self.target_mask_good], self.pim_all_bad[self.target_mask_bad]))
        print(pim_ordered)
        np.savetxt('ostar-pims-ordered.txt', pim_ordered)
        ebv_comb = np.concatenate((ebv_good, ebv_bad))
        flux_all = np.concatenate((self.flux_good, self.flux_bad))
        print(len(ebv_good))
        print(len(ebv_bad))
        print('\nPlotting spectra...')
        plt.figure(figsize=(8.27, 11.69))
        for c, k in enumerate(np.arange(len(ebv_comb))):
            plt.subplot(14,5,k+1)
            if c >= len(ebv_good):  # plot bad spec in different colour
                plt.plot(self.wave, self.dered(flux_all[k], ebv_comb[k]), linewidth=0.5, c='b')
            else:
                plt.plot(self.wave, self.dered(flux_all[k], ebv_comb[k]), linewidth=0.5, c='k')
            plt.ylabel('Flux', fontsize=5, labelpad=0.5)
            plt.xlabel(r'$Wavelength, \AA$', fontsize=5, labelpad=0.2)
            plt.tick_params(axis='both', which='both', labelsize=1, length=0, direction='in', labelleft=False, labelbottom=False)
        plt.subplots_adjust(wspace=0.15, hspace=0.2)
        #plt.tight_layout()
        plt.savefig('/home/lewishill/Documents/MaStar/jupyter/carbon_paper/plots/C-star-grid-portrait.png', bbox_inches='tight', dpi=600)

    def plot_Cdwarfs(self):
        header = fits.open('/home/lewishill/Downloads/mastarall_v1-7-7-ALLVISITS-PIM.fits')
        ebv_all = header[1].data['ebv']
        pim_allspec = header[1].data['pim']
        ebv_good = [ebv_all[np.where(pim_allspec == i)] for i in self.pim_all_good[self.target_mask_good]]
        ebv_bad = [ebv_all[np.where(pim_allspec == i)] for i in self.pim_all_bad[self.target_mask_bad]]
        print(ebv_bad, ebv_good)
        ebv_comb = ebv_good
        flux_all = np.concatenate((self.flux_good, self.flux_bad))

        print('\nPlotting spectra...')
        plt.figure(figsize=(8.27, 4))
        for k in np.arange(len(ebv_comb)):
            plt.subplot(2,2,k+1)
            plt.plot(self.wave, self.dered(flux_all[k], ebv_comb[k]), linewidth=0.5, c='k')
            plt.ylabel('Relative flux', fontsize=10)
            plt.xlabel(r'Wavelength, $\AA$', fontsize=10)
            plt.tick_params(axis='both', which='major', labelsize=8, length=3, labelleft=False, labelbottom=True, direction='in')
        # plt.subplots_adjust(wspace=0.15, hspace=0.2)
        plt.tight_layout()
        plt.savefig('/home/lewishill/Documents/MaStar/jupyter/carbon_paper/plots/Dwarf-C-star-grid.png', bbox_inches='tight', dpi=600)

    def plot_subgrid(self):
        header = fits.open('/home/lewishill/Downloads/mastarall_v1-7-7-ALLVISITS-PIM.fits')
        ebv_all = header[1].data['ebv']
        pim_allspec = header[1].data['pim']
        ebv_good = [ebv_all[np.where(pim_allspec == i)] for i in self.pim_all_good[self.target_mask_good]]
        ebv_bad = [ebv_all[np.where(pim_allspec == i)] for i in self.pim_all_bad[self.target_mask_bad]]
        print(ebv_bad, ebv_good)
        ebv_comb = np.concatenate((ebv_good, ebv_bad))
        flux_all = np.concatenate((self.flux_good, self.flux_bad))
        print('\nPlotting spectra...')
        plt.figure(figsize=(8.27, 4))
        for i, k in zip([0,1,2,3,4,5], [0,1,5,2,3,4]):# np.arange(len(ebv_comb)):
            plt.subplot(2,3,i+1)
            plt.plot(self.wave, self.dered(flux_all[k], ebv_comb[k]), linewidth=0.5, c='k')
            plt.ylabel('Relative flux', fontsize=14)
            plt.xlabel(r'Wavelength, $\AA$', fontsize=14)
            plt.tick_params(axis='both', which='major', labelsize=12, length=3, labelleft=False, labelbottom=True, direction='in')
        # plt.subplots_adjust(wspace=0.15, hspace=0.2)
        plt.tight_layout()
        plt.savefig('/home/lewishill/Documents/MaStar/jupyter/carbon_paper/plots/C-O-example-grid.png', bbox_inches='tight', dpi=600)

    def get_targets(self, which):
        if which == 'bad':
            x = []
            for i in self.pims_todo:
                if i in self.pim_all_bad:
                    x.append(np.where(self.pim_all_bad == i)[0][0])
            return x
        if which == 'good':
            x = []
            for i in self.pims_todo:
                if i in self.pim_all_good:
                    x.append(np.where(self.pim_all_good == i)[0][0])
            return x

    def dered(self, flux_to_dered, ebv, a_v=None, r_v=3.1, model='f99'):
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
        if (a_v is None) and (ebv is None):
            raise ValueError('Must specify either a_v or ebv')
        if (a_v is not None) and (ebv is not None):
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

        reddening_curve = 10 ** (0.4 * ebv * (k + r_v))

        corrected_flux = flux_to_dered * reddening_curve

        return corrected_flux

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

            ivar_ = self.ivar  # get inverse variance
            ivar_x = np.arange(len(ivar_))
            idx = np.nonzero(ivar_)  # nonzero values in ivar
            f = interp1d(ivar_x[idx], ivar_[idx], fill_value='extrapolate')  # interp function with non zero values
            ivar_new = f(ivar_x)  # interpolate where zero values occur
            if np.any(ivar_new < 0):    # sometime nans appear at end of error array. This fills them forward.
                ivar_fill_value = ivar_new[np.where(ivar_new<0)[0][0]-1]
                for i in np.where(ivar_new < 0)[0]:
                    ivar_new[i] = ivar_new[np.where(ivar_new < 0)[0][0] - 1]
            sd = ivar_new ** -0.5  # change inverse variance to standard error
            sd_pcnt = sd / flux_new  # error as a percentage of the flux

            self.corrected_flux_med = corrected_flux / np.median(corrected_flux)  # median normalise
            self.yerr = self.corrected_flux_med * sd_pcnt
        else:
            ivar_ = self.ivar  # get inverse variance
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

class point_estimates:
    """Use the sampler output to calculate point estimate parameters and their errors."""

    def __init__(self, spec_info, samples, pim, model):
        self.params = {}
        self.chi = {}
        self.ppxf_fit = {}
        self.pim = pim
        self.model = model
        self.samples_clean = samples.reshape((-1, var.ndim))

        # data from 'clean_spec' instance.
        self.mast_flux = spec_info.corrected_flux_med
        self.yerr = spec_info.yerr

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

    def params_err(self, teff='median', logg='mode', zh='mode', alpha='mode'):
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
        self.params[self.model + '_params'] = params_temp

    def get_model_fit(self):
        """Get polynomial corrected model fit from pPXF."""
        if var.alpha:
            model_flux = model_spec([self.params[self.model+'_params'][0][0], self.params[self.model+'_params'][1][0],
                                     self.params[self.model+'_params'][2][0], self.params[self.model+'_params'][3][0]],
                                    self.model)
            sol = (ppxf(model_flux, self.mast_flux, noise=self.yerr, velscale=var.velscale, start=var.start, degree=-1,
                        mdegree=var.mdegree, moments=var.moments, quiet=True))
            bestfit_model = sol.bestfit
            self.ppxf_fit[self.model] = bestfit_model      # append fit to the ppxf fit dictionary
            return self.ppxf_fit
        else:
            model_flux = model_spec([self.params[self.model+'_params'][0][0], self.params[self.model+'_params'][1][0],
                                     self.params[self.model+'_params'][2][0]], self.model)
            sol = (ppxf(model_flux, self.mast_flux, noise=self.yerr, velscale=var.velscale, start=var.start, degree=-1,
                        mdegree=var.mdegree, moments=var.moments, quiet=True))
            bestfit_model = sol.bestfit
            self.ppxf_fit[self.model] = bestfit_model  # append fit to the ppxf fit dictionary
            return self.ppxf_fit

    def get_chi2(self):
        """Get chi2 based on the point estimate parameters and append to chi dictionary for later comparison."""
        chi = np.sum(((self.mast_flux - self.ppxf_fit[self.model]) ** 2 / self.yerr ** 2)) / (len(self.mast_flux
                                                                                                  - var.ndim))
        self.params[self.model + '_chi'] = chi
        return self.chi


class plotting:
    """A class to retrospectively plot results from mcmc run"""
    def __init__(self, point_estimates, clean_spec, samples, pim, model, output_folder):
        self.model = model
        self.pim = pim
        self.clean_spec = clean_spec
        self.samples_cut = samples
        self.output_folder = output_folder
        self.point_estimates = point_estimates
        # plotting.trace(self)    # create trace plot and save
        # plotting.bestfit(self)   # create bestfit plot and save
        plotting.corner(self)   # create corner plot and save


    def trace(self):
        """Plot the trace for each parameter."""

        plt.figure(figsize=(16, 30))
        plt.subplot(var.ndim, 1, 1)
        plt.plot(self.samples_cut[:,:,0].T, '--', color='k', alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.ylabel('Effective Temperature (Kelvin)', fontsize=16)
        plt.xlabel('Iterations', fontsize=16)

        plt.subplot(var.ndim, 1, 2)
        plt.plot(self.samples_cut[:,:,1].T, '--', color='k', alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.ylabel('Log g', fontsize=16)
        plt.xlabel('Iterations', fontsize=16)

        plt.subplot(var.ndim, 1, 3)
        plt.plot(self.samples_cut[:,:,2].T, '--', color='k', alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.ylabel('Metallicity ([Fe/H])', fontsize=16)
        plt.xlabel('Iterations', fontsize=16)

        plt.subplot(var.ndim, 1, 4)
        plt.plot(self.samples_cut[:,:,3].T, '--', color='k', alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.ylabel(r'Alpha abundance $([\alpha/Fe])$', fontsize=16)
        plt.xlabel('Iterations', fontsize=16)

        plt.tight_layout()
        plt.savefig(self.output_folder + str(self.pim) + self.model + '_trace.png', bbox_inches='tight')
        plt.close()

    def bestfit(self):
        """Plot bestfit for model spec"""
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 12), sharey=False, sharex=True)
        axes.set_title('MaStar PIM: ' + str(self.pim), fontsize=32)
        divider = make_axes_locatable(axes)
        ax2 = divider.append_axes("bottom", size="36%", pad=0)
        axes.figure.add_axes(ax2)
        axes.plot(self.clean_spec.wave, self.clean_spec.corrected_flux_med, 'k', linewidth=3, label='MaStar spectrum')
        if self.model == 'BOSZ':
            model_fit = self.point_estimates.ppxf_fit['BOSZ']
        elif self.model == 'MARCS':
            model_fit = self.point_estimates.ppxf_fit['MARCS']
        axes.plot(self.clean_spec.wave, model_fit, c='b', lw=2, label=self.model+' model')
        # if len(self.point_estimates.ppxf_fit) == 2:      # check if there's a MARCS solution
        #     axes.plot(self.clean_spec.wave, self.point_estimates.ppxf_fit['MARCS'], c='r', lw=2,
        #               label='MARCS model')
        axes.legend(loc=0, fontsize=20)
        axes.set_ylabel('Relative flux', fontsize=25)
        axes.tick_params(axis='both', which='major', labelsize=20, direction='in')

        ax2.plot(self.clean_spec.wave, self.clean_spec.corrected_flux_med - model_fit, c='b', lw=2)
        # if len(self.point_estimates.ppxf_fit) == 2:      # check if there's a MARCS solution
        #     ax2.plot(self.clean_spec.wave, self.clean_spec.corrected_flux_med - self.point_estimates.ppxf_fit['MARCS']
        #             , c='r', lw=2)
        ax2.set_ylabel('Residual flux', fontsize=25)
        ax2.set_xlabel(r'Wavelength, $\AA$', fontsize=25)
        ax2.tick_params(axis='both', which='major', labelsize=20, direction='in')
        ax2.axhline(y=0, c='r', linestyle='--', lw=2)
        axes.set_xticks([])
        plt.tight_layout()
        plt.savefig(self.output_folder + str(self.pim) + self.model + '_bestfit.png', bbox_inches='tight')
        plt.close()

    def corner(self):
        """Plot corner."""
        plt.clf()
        if var.alpha:
            fig = corner(self.point_estimates.samples_clean, labels=["$T_{\mathrm{eff}}$", "log $g$", "[Fe/H]", r"[$\alpha$/Fe]"],
                         quantiles=None, show_titles=False, title_kwargs={"fontsize": 22},
                         label_kwargs={"fontsize": 24}, color='dodgerblue', labelpad=0.1)
        else:
            fig = corner(self.point_estimates.samples_clean, labels=["$T_{\mathrm{eff}}$", "$log g$", "$[Fe/H]$"],
                         quantiles=None, show_titles=False, title_kwargs={"fontsize": 22},
                         label_kwargs={"fontsize": 24}, color='dodgerblue', labelpad=20)
        axes = np.array(fig.axes).reshape((var.ndim, var.ndim))   # Extract the axes
        for i in range(var.ndim):       # Loop over the diagonal
            ax = axes[i, i]
            if i == 0:
                ax.set_title('$%d^{+%d}_{-%d}$K' % (
                    np.round(self.point_estimates.params[self.model+'_params'][i][0], 0),
                    np.round(self.point_estimates.params[self.model+'_params'][i][2], 0),
                    np.round(self.point_estimates.params[self.model+'_params'][i][1], 0)), fontsize=22)
            if i == 1:
                ax.set_title('$%6.2f^{+%6.2f}_{-%6.2f}$' % (
                    np.round(self.point_estimates.params[self.model+'_params'][i][0], 2),
                    np.round(self.point_estimates.params[self.model+'_params'][i][2], 2),
                    np.round(self.point_estimates.params[self.model+'_params'][i][1], 2)), fontsize=22)
            if i == 2:
                ax.set_title('$%6.2f^{+%6.2f}_{-%6.2f}$' % (
                    np.round(self.point_estimates.params[self.model+'_params'][i][0], 2),
                    np.round(self.point_estimates.params[self.model+'_params'][i][2], 2),
                    np.round(self.point_estimates.params[self.model+'_params'][i][1], 2)), fontsize=22)
            if i == 3:
                ax.set_title('$%6.2f^{+%6.2f}_{-%6.2f}$' % (
                    np.round(self.point_estimates.params[self.model+'_params'][i][0], 2),
                    np.round(self.point_estimates.params[self.model+'_params'][i][2], 2),
                    np.round(self.point_estimates.params[self.model+'_params'][i][1], 2)), fontsize=22)
            ax.axvline(self.point_estimates.params[self.model+'_params'][i][0], color="r")
            ax.axvline(self.point_estimates.params[self.model+'_params'][i][0] -
                       self.point_estimates.params[self.model+'_params'][i][1], color="k", linestyle='--')
            ax.axvline(self.point_estimates.params[self.model+'_params'][i][0] +
                       self.point_estimates.params[self.model+'_params'][i][2], color="k", linestyle='--')

        for yi in range(var.ndim):      # Loop over the histograms
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(self.point_estimates.params[self.model+'_params'][xi][0], color="r")
                ax.axhline(self.point_estimates.params[self.model+'_params'][yi][0], color="r")
                ax.plot(self.point_estimates.params[self.model+'_params'][xi][0],
                        self.point_estimates.params[self.model+'_params'][yi][0], "sr")

        for ax in axes[1:, 0]:
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16)
            formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(formatter)
        for ax in axes[-1, :]:
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(16)
            formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
            ax.xaxis.set_major_formatter(formatter)

        plt.tight_layout()
        plt.savefig(self.output_folder + str(self.pim) + self.model + '_corner.png', bbox_inches='tight')
        plt.close()

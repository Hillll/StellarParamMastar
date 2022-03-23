
class spd_setup:
    """
    Define the input parameters and directories for input/output.
    """
    def __init__(self):
        # directories
        self.data_direc = '/home/lewishill/PycharmProjects/SPD/input/'
        self.output_direc = '/home/lewishill/PycharmProjects/SPD/output_interpLH/'
        self.plots_output_direc = '/home/lewishill/PycharmProjects/SPD/plots/interp_LH/'
        self.output_description = 'Test.'

        self.spec_file = 'mastar-goodspec-v3_1_1-v1_7_7.fits'
        self.est_file = 'mastar-goodspec-mpl11-gaia-DT-v2-mpl11-v3_1_1-v1_7_5.fits'
        self.nums_file = 'nums_non_contiguous.npz'   # ndarray of indices to do from self.spec_file

        # input options
        self.alpha = True
        self.min_lambda = 3800  # set wavelength lower limit. If -999 then minimum lambda is used.
        self.max_lambda = 5000  # set wavelength upper limit. If -999 then maximum lambda is used.

        # output options
        self.plot = True    # Whether to plot the results
        self.save_chains = False    # To save the MCMC chains
        self.save_params = True     # To save the output params

        # mcmc params
        self.early_stopping = True   # whether to measure convergence and stop when deemed comverged
        if self.alpha:
            self.ndim = 4
        else:
            self.ndim = 3
        self.nwalkers = 20  # N MCMC walkers
        self.burnin = 100   # Length of burn in iterations
        self.niter = 100    # Length of iterations that will be used in the posterior
        self.a = 5          # Emcee acceptance fraction of walkers
        self.progress = True  # view progress in chains

        # pPXF params
        self.velscale = 69  # for SDSS spectra
        self.start = [0, 10]  # initial guess for velocities
        self.mdegree = 6    # Multiplicative polynomial order
        self.moments = 2    # Fit for V and sigma
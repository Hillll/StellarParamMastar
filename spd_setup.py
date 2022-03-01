
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
        self.nums_file = 'nums_todo.npz'   # ndarray of monotonically increasing values

        # input options
        self.alpha = True

        # output options
        self.plot = True
        self.save_chains = False
        self.save_params = True

        # mcmc params
        self.early_stopping = True   # whether to measure convergence and stop when deemed comverged
        if self.alpha:
            self.ndim = 4
        else:
            self.ndim = 3
        self.nwalkers = 20
        self.burnin = 200
        self.niter = 2000
        self.a = 5
        self.progress = True  # view progress in chains

        # pPXF params
        self.velscale = 69  # for SDSS spectra
        self.start = [0, 10]  # initial guess for velocities
        self.mdegree = 6
        self.moments = 2
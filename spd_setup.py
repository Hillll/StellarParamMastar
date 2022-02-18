
class spd_setup:
    """
    Define the input parameters and directories for input/output.
    """
    def __init__(self):
        # directories
        self.data_direc = '/home/lewishill/PycharmProjects/SPD/input/'
        self.output_direc = '/home/lewishill/PycharmProjects/SPD/output/'
        self.plots_output_direc = '/home/lewishill/PycharmProjects/SPD/plots/'

        self.spec_file = 'mastar-goodspec-v3_1_1-v1_7_7.fits'
        self.est_file = 'mastar-goodspec-mpl11-gaia-DT-v2-mpl11-v3_1_1-v1_7_5.fits'
        self.nums_file = 'nums_todo-temp.npz'   # ndarray of monotonically increasing values

        # input options
        self.alpha = True

        # output options
        self.plot = True
        self.save_chains = False
        self.save_params = True

        # mcmc params
        if self.alpha:
            self.ndim = 4
        else:
            self.ndim = 3
        self.nwalkers = 20
        self.burnin = 100
        self.niter = 3000
        self.a = 5

        # pPXF params
        self.velscale = 69  # for SDSS spectra
        self.start = [0, 10]  # initial guess for velocities
        self.mdegree = 6
        self.moments = 2
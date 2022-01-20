
class spd_setup:
    """
    Define the input parameters and directories for input/output.
    """
    def __init__(self):
        # directories
        self.data_direc = '/home/lewishill/Downloads/'
        self.output_direc = '/mnt/lustre/lhill/shera/mcmc/spd/output/'

        self.spec_file = 'mastar-goodspec-v3_1_1-v1_7_7.fits'
        self.est_file = 'mastar-goodspec-mpl11-gaia-DT-v2-mpl11-v3_1_1-v1_7_5.fits'
        self.nums_file = 'nums_todo.npz'   # ndarray of monotonically increasing values
        

        self.alpha = False

        # mcmc params
        if self.alpha:
            self.ndim = 4
        else:
            self.ndim = 3
        self.nwalkers = 10
        self.niter = 30
        self.burnin = 30
        self.a = 5

        # pPXF params
        self.velscale = 69  # for SDSS spectra
        self.start = [0, 10]  # initial guess for velocities
        self.mdegree = 6
        self.moments = 2
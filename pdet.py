import numpy as np
import scipy
from tqdm.auto import tqdm
import multiprocessing
import gwdet
from astropy.cosmology import Planck18

p = gwdet.detectability()

file = np.load("data/pdet_nsamples_5e2_(attempt_2).npz")
m1grid, m2grid, zgrid, pdet_for_interpolant = [file[key] for key in file.files]

pdet_interpolant = scipy.interpolate.RegularGridInterpolator((m1grid, m2grid, zgrid), pdet_for_interpolant,
                                                             bounds_error=False, fill_value=None,
                                                             method='linear')

# Override gwdet probability calculator with our own
def p(m1, m2, z):
    pdet = pdet_interpolant(np.array([m1, m2, z]).T)
    return pdet[0] if len(pdet) == 1 else pdet

# Returns the indexes of the selected cell and the median of p_det in the cell computed varying
# one of the mass component of the binary
def p_det_median_cell(_m, m, _z):
    return np.median(p(np.zeros(len(m)) + _m, m, np.zeros(len(m)) + _z))

# Support function for multiprocessing
def _p_det_median_cell_worker(args):
    i, j, _m, m, _z = args
    return i, j, p_det_median_cell(_m, m, _z)

# Returns the medians of p_det computed on a given (m, z) grid
# p_det[z_index, m_index]
def p_det_median(m, z):
    # Initialize the result variable
    p_det = np.zeros((len(z), len(m)))

    # Initialize the list of arguments to be passed to the computation function
    args = [(i, j, _m, m, _z) for (i, _z) in enumerate(z) for (j, _m) in enumerate(m)]

    # Use unordered multiprocessing for the computation
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(_p_det_median_cell_worker, args), total=len(args),
                            desc='Commputing median p_det for each (m, z) pair'))
        
    # Reorder and return the results
    for i, j, res in results:
        p_det[i, j] = res

    return p_det

# Computes the spacetime-volume of a (m, z) cell assuming a uniform population in m and z using montecarlo integration
def VT_pop_uniform_cell(T_obs, z_min, z_max, m_min, m_max, cell_grid_dim=100, mc_n_samples=10000):
    # Mass cell over which p_det_median is computed
    m_cell_grid = np.linspace(1, 100, cell_grid_dim)
    # Uniform sampling of m and z in the cell
    m_samples = np.random.uniform(m_min, m_max, mc_n_samples)
    z_samples = np.random.uniform(z_min, z_max, mc_n_samples)

    # Compute the comsological part using vectorized function calls
    # The differential_comoving_volume function returns the result per unit solid angle
    dcomv = 1 / (1 + z_samples) * 4 * np.pi * Planck18.differential_comoving_volume(z_samples).to_value()

    # Compute p_det_median
    f = lambda m, z: np.median(p(np.zeros(cell_grid_dim) + m, m_cell_grid, np.zeros(cell_grid_dim) + z))
    p_det_median = np.array([f(m, z) for m, z in zip(m_samples, z_samples)])
    
    # Montecarlo integration
    VT = T_obs * (z_max - z_min) * (m_max - m_min) * np.mean(dcomv * p_det_median)

    return VT

# Utility function for multiprocessing
def _VT_pop_uniform_cell_worker(args):
    i, j, T_obs, z_min, z_max, m_min, m_max, cell_grid_dim, mc_n_samples = args
    return i, j, VT_pop_uniform_cell(T_obs, z_min, z_max, m_min, m_max, cell_grid_dim, mc_n_samples)

# Compute VT for each cell in a (m, z) grid
def VT_pop_uniform(T_obs, z, m, cell_grid_dim=100, mc_n_samples=2000):
    # Get lower and upper boundaries of the cells
    z_left, z_right = z[:-1], z[1:]
    m_left, m_right = m[:-1], m[1:]

    # Initialize the result variable
    VT = np.zeros((len(z) - 1, len(m) - 1))

    # Initialize the list of arguments to be passed to the computation function
    args = [(i, j, T_obs, z_l, z_r, m_l, m_r, cell_grid_dim, mc_n_samples)
            for (i, (z_l, z_r)) in enumerate(zip(z_left, z_right)) for (j, (m_l, m_r)) in enumerate(zip(m_left, m_right))]
    
    # Use unordered multiprocessing for the computation
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(_VT_pop_uniform_cell_worker, args), total=len(args),
                            desc='Computing VT for each (m, z) pair for a uniformly distributed population of sources'))
    
    # Reorder and return the results
    for i, j, res in results:
        VT[i, j] = res

    return VT

# Computes the spacetime-volume of a (m, z) cell assuming a uniform population in m and z
# using montecarlo integration for a fixed mass ratio q = m2 / m1, where m2 <= m1
def VT_pop_uniform_cell_q(T_obs, z_min, z_max, m_min, m_max, q, mc_n_samples=10000):
    # Uniform sampling of m1 and z in the cell. m2 = m1 * q
    m1_samples = np.random.uniform(m_min, m_max, mc_n_samples)
    m2_samples = m1_samples * q
    z_samples = np.random.uniform(z_min, z_max, mc_n_samples)

    # Compute the comsological part using vectorized function calls
    # The differential_comoving_volume function returns the result per unit solid angle
    dcomv = 1 / (1 + z_samples) * 4 * np.pi * Planck18.differential_comoving_volume(z_samples).to_value()

    # Compute p_det for each sample
    p_det = p(m1_samples, m2_samples, z_samples)

    # Montecarlo integration
    VT = T_obs * (z_max - z_min) * (m_max - m_min) * np.mean(dcomv * p_det)

    return VT

# Utility function for multiprocessing
def _VT_pop_uniform_cell_q_worker(args):
    i, j, T_obs, z_min, z_max, m_min, m_max, q, mc_n_samples = args
    return i, j, VT_pop_uniform_cell_q(T_obs, z_min, z_max, m_min, m_max, q, mc_n_samples)

# Compute VT for each cell in a (m, z) grid, for a fixed mass ratio
def VT_pop_uniform_q(T_obs, z, m1, q, mc_n_samples=2000):
    # Get lower and upper boundaries of the cells
    z_left, z_right = z[:-1], z[1:]
    m1_left, m1_right = m1[:-1], m1[1:]

    # Initialize the result variable
    VT = np.zeros((len(z) - 1, len(m1) - 1))

    # Initialize the list of arguments to be passed to the computation function
    args = [(i, j, T_obs, z_l, z_r, m1_l, m1_r, q, mc_n_samples)
            for (i, (z_l, z_r)) in enumerate(zip(z_left, z_right)) for (j, (m1_l, m1_r)) in enumerate(zip(m1_left, m1_right))]
    
    # Use unordered multiprocessing for the computation
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(_VT_pop_uniform_cell_q_worker, args), total=len(args),
                            desc='Computing VT for each (m, z) pair for a uniformly distributed population of sources with fixed q = {0}'.format(q)))
    
    # Reorder and return the results
    for i, j, res in results:
        VT[i, j] = res

    return VT

# A class to describe a mass-redshift bin
class MZbin:
    def __init__(self, minf, msup, zinf, zsup):
        self.minf = minf
        self.msup = msup
        self.zinf = zinf
        self.zsup = zsup

# Child of the MZbin class. This bin considers only binaries with a fixed mass ratio q
class MZQbin(MZbin):
    def __init__(self, minf, msup, zinf, zsup, q): # q = m2 / m1, with m2 <= m1
        super().__init__(minf, msup, zinf, zsup)
        self.q = q
    
    # Compute VT for the bin using montecarlo integration
    def VTmc(self, Tobs, mcn=100000):
        self.Tobs = Tobs
        # from pdet import VT_pop_uniform_cell_q
        self.VT = VT_pop_uniform_cell_q(Tobs, self.zinf, self.zsup, self.minf, self.msup, self.q, mcn)
        return self.VT
    
    # Computes the confidence interval for the countings ratio using the LRT statistics (see bin_comparison_2)
    def alphaLRT(self, bin2, cl, R1_R2, N2, Tobs=1, mcn=100000):
        lambda_LR_target = scipy.stats.chi2.isf(cl, 1)

        for bin in (self, bin2):
            if not hasattr(bin, 'VT'):
                bin.VTmc(Tobs, mcn)
        a = bin2.VT / self.VT
        N1_N2_0_ref = R1_R2 / a

        alpha_eq = lambda x, N_2, alpha_0_ref: ((np.exp(lambda_LR_target / (2 * N_2)) *\
                                                 (1 + 1 / x) ** x * (1 + x)) ** (1 / (1 + x))) * alpha_0_ref ** (x / (1 + x)) - alpha_0_ref - 1
        from scipy.optimize.elementwise import find_root
        if np.isscalar(N2):
            N2 = np.array([N2])
        # Remember that you might want to make the interval searched by the solver wider
        alpha_target_sx = np.array([find_root(alpha_eq, (0.01, N1_N2_0_ref), args=(n_2, N1_N2_0_ref)).x for n_2 in N2])
        alpha_target_dx = np.array([find_root(alpha_eq, (N1_N2_0_ref, 100), args=(n_2, N1_N2_0_ref)).x for n_2 in N2])

        return alpha_target_sx, alpha_target_dx, N1_N2_0_ref, a
    
    # Returns samples of the countings ratio using the fact that
    # the countings ratio is distributed following a beta prima distribution
    # See https://en.wikipedia.org/wiki/Beta_prime_distribution#Generalization for details
    def alphaBayesHist(self, bin2, R1_R2, N2, Tobs=1, mcn=100000, alpha_prior=0.5, beta_prior=0, nsamples=100000):
        for bin in (self, bin2):
            if not hasattr(bin, 'VT'):
                bin.VTmc(Tobs, mcn)
        a = bin2.VT / self.VT
        N1_N2 = R1_R2 / a
        N1 = N1_N2 * N2

        alpha1, beta1 = alpha_prior + N1, beta_prior + 1
        alpha2, beta2 = alpha_prior + N2, beta_prior + 1

        l1_l2_samples = scipy.stats.betaprime.rvs(alpha1, alpha2, scale=beta2/beta1, size=nsamples)

        return l1_l2_samples, N1_N2, a
    
    # Computes the confidence interval for the countings ratio using the fact that
    # the countings ratio is distributed following a beta prima distribution
    def alphaBayesCI(self, interval, bin2, R1_R2, N2, Tobs=1, mcn=100000, alpha_prior=0.5, beta_prior=0):
        for bin in (self, bin2):
            if not hasattr(bin, 'VT'):
                bin.VTmc(Tobs, mcn)
        a = bin2.VT / self.VT
        N1_N2 = R1_R2 / a
        N1 = N1_N2 * N2

        alpha1, beta1 = alpha_prior + N1, beta_prior + 1
        alpha2, beta2 = alpha_prior + N2, beta_prior + 1

        median = scipy.stats.betaprime.median(alpha1, alpha2, scale=beta2/beta1)
        CI = scipy.stats.betaprime.interval(interval, alpha1, alpha2, scale=beta2/beta1)

        return (CI[0], median, CI[1]), N1_N2, a
    
    # Computes the confidence interval for the rate ratio using the fact that
    # the countings ratio is distributed following a beta prima distribution
    # The rate ratio is obtained multiplying the latter by the VT ratio.
    # See https://en.wikipedia.org/wiki/Beta_prime_distribution#Properties for details
    def RateRatioBayesCI(self, interval, bin2, N1_N2, N2, Tobs=1, mcn=100000, alpha_prior=0.5, beta_prior=0):
        for bin in (self, bin2):
            if not hasattr(bin, 'VT'):
                bin.VTmc(Tobs, mcn)
        a = bin2.VT / self.VT
        N1 = N1_N2 * N2
        R1_R2 = N1_N2 * a

        alpha1, beta1 = alpha_prior + N1, beta_prior + 1
        alpha2, beta2 = alpha_prior + N2, beta_prior + 1

        median = scipy.stats.betaprime.median(alpha1, alpha2, scale=beta2/beta1 * a)
        CI = scipy.stats.betaprime.interval(interval, alpha1, alpha2, scale=beta2/beta1 * a)

        return (CI[0], median, CI[1]), R1_R2, a
    
# Utility function to initilize bins using multiprocessing  
def _bin_initializer_worker(args):
    i, bin, T_obs, mcn = args
    bin.VTmc(T_obs, mcn)
    return i, bin

# Computes the confidence interval for the counting ratio on a (R1_R2, bin_z) grid
# using a reference bin2_ref having a reference number of events N2_ref.
# Note that VT is computed only if bins have not been initialized
def alphaBayesCI_map(interval, bin2_ref, R1_R2_axis, bins_z_axis, N2_ref, Tobs=1, mcn=10000, alpha_prior=0.5, beta_prior=0):
    # Initialize the result variable as a list (use list comprehension)
    alpha_CI = [[None for _ in range(len(bins_z_axis))] for _ in range(len(R1_R2_axis))]

    with tqdm(total=len(R1_R2_axis) * len(bins_z_axis), desc='Computing number of events ratio map') as pbar:
        for i, R1_R2 in enumerate(R1_R2_axis):
            for j, bin_z in enumerate(bins_z_axis):
                alpha_CI[i][j] = bin_z.alphaBayesCI(interval, bin2_ref, R1_R2, N2_ref, Tobs, mcn, alpha_prior, beta_prior)
                pbar.update(1)

    return alpha_CI

# Computes the confidence interval for the rate ratio on a (N1_N2, bin_z) grid
# using a reference bin2_ref having a reference number of events N2_ref.
# Note that VT is computed only if bins have not been initialized
def RateRatioBayesCI_map(interval, bin2_ref, N1_N2_axis, bins_z_axis, N2_ref, Tobs=1, mcn=10000, alpha_prior=0.5, beta_prior=0):
    # Initialize the result variable as a list (use list comprehension)
    R_CI = [[None for _ in range(len(bins_z_axis))] for _ in range(len(N1_N2_axis))]

    with tqdm(total=len(N1_N2_axis) * len(bins_z_axis), desc='Computing rate ratio map') as pbar:
        for i, N1_N2 in enumerate(N1_N2_axis):
            for j, bin_z in enumerate(bins_z_axis):
                R_CI[i][j] = bin_z.RateRatioBayesCI(interval, bin2_ref, N1_N2, N2_ref, Tobs, mcn, alpha_prior, beta_prior)
                pbar.update(1)

    return R_CI
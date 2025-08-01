import numpy as np
from tqdm.auto import tqdm
import multiprocessing
import gwdet
from astropy.cosmology import Planck18

p = gwdet.detectability()

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

# Support function for multiprocessing
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
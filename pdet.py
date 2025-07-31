import numpy as np
from tqdm.auto import tqdm
import multiprocessing
import gwdet

p = gwdet.detectability()

def __p_det_median_cell__(args):
    i, j, _m, m, _z, m_len = args
    # Returns the indexes of the selected cell and the median of p_det in the cell computed varying
    # one of the mass component of the binary
    return i, j, np.median(p(np.zeros(m_len) + _m, m, np.zeros(m_len) + _z))

def p_det_median(m, z, file=None, save=False):
    # Save the length of m and z
    m_dim, z_dim = len(m), len(z)

    # Initialize the result variable
    p_det = np.zeros((z_dim, m_dim))

    # Initialize the list of arguments to be passed to the computation function
    args = [(i, j, _m, m, _z, m_dim) for (i, _z) in enumerate(z) for (j, _m) in enumerate(m)]

    # Use unordered multiprocessing for the computation
    num_processes = multiprocessing.cpu_count() # Number of CPUs
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(__p_det_median_cell__, args), total=len(args),
                            desc='Commputing median p_det for each (m, z) pair'))
        
    # Reorder and return the results
    for i, j, res in results:
        p_det[i, j] = res

    return p_det
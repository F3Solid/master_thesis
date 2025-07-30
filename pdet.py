import numpy as np
from tqdm.auto import tqdm
import multiprocessing
import gwdet

p = gwdet.detectability()

def __p_det_median_cell__(args):
    i, j, _m, m, _z, m_len = args
    return i, j, np.median(p(np.zeros(m_len) + _m, m, np.zeros(m_len) + _z))

def p_det_median(m, z):
    m_dim = len(m)
    z_dim = len(z)

    p_det = np.zeros((z_dim, m_dim))
    
    args = [(i, j, _m, m, _z, m_dim) for (i, _z) in enumerate(z) for (j, _m) in enumerate(m)]

    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(__p_det_median_cell__, args), total=len(args), desc='Commputing median p_det for each (m, z) pair'))

    for i, j, res in results:
        p_det[i, j] = res

    return p_det
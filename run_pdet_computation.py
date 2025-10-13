import numpy as np
from pdet_gwbench import *

z_max_m_min = ((0.1, 1), (1, 30), (2, 85))
z_min_m_max = ((6, 10), (8, 15))

old_grids = np.load("data/pdet_nsamples_5e2_(attempt_3).npz")

m1grid = np.linspace(50, 100, 51)
m2grid = np.copy(m1grid)
zgrid = np.copy(old_grids["zgrid"])

meshcoord, meshgrid, pdet_for_interpolant = initialize_pdet_grids((m1grid, m2grid, zgrid), z_max_m_min, z_min_m_max)
grids = merge_pdet_grids((m1grid, m2grid, zgrid, pdet_for_interpolant), "data/pdet_nsamples_5e2_(attempt_3).npz")
old_grids.close()
meshcoord, meshgrid, pdet_for_interpolant = initialize_pdet_grids((grids["m1grid"], grids["m2grid"], grids["zgrid"]), z_max_m_min, z_min_m_max, pdet_for_interpolant=grids["pdet_for_interpolant"])

meshvalues = compute_pdet_for_interpolant(meshgrid, meshcoord, grids["m1grid"], grids["m2grid"], grids["zgrid"], grids["pdet_for_interpolant"],
                                          n_sample=5e2,
                                          save_checkpoint=True,
                                          save_step=-10,
                                          checkpoint_file_path="data/pdet_computation_checkpoint.npz",
                                          from_checkpoint=True)

file = np.load("data/pdet_computation_checkpoint.npz")

m1grid = file['m1grid']
m2grid = file['m2grid']
zgrid = file['zgrid']
pdet_for_interpolant = file['pdet_for_interpolant']
meshcoord = file['meshcoord']
meshvalues = file['meshvalues']

file.close()

for ijk, val in zip(meshcoord, meshvalues):
    i, j, k = ijk
    pdet_for_interpolant[i, j, k] = val
    pdet_for_interpolant[j, i, k] = val

np.savez("data/pdet_nsamples_5e2_(attempt_4)", m1grid=m1grid, m2grid=m2grid, zgrid=zgrid, pdet_for_interpolant=pdet_for_interpolant)
import numpy as np
from comp_pdet import pdet
from tqdm.auto import tqdm
from joblib import Parallel, delayed, cpu_count
import logging

z_max_m_min = ((0.1, 1), (1, 30))
z_min_m_max = ((6, 10),)

m1grid = np.concatenate((np.linspace(1, 5, 15, endpoint=False),
                         np.linspace(5, 15, 30, endpoint=False),
                         np.linspace(15, 30, 15, endpoint=False),
                         np.linspace(30, 40, 30, endpoint=False),
                         np.linspace(40, 50, 10)))
m2grid = np.copy(m1grid)
zgrid = np.concatenate((np.geomspace(1e-4, 1e-1, 200, endpoint=False),
                        np.geomspace(1e-1, 1, 50, endpoint=False),
                        np.geomspace(1, 10, 50)))

grids = (m1grid, m2grid, zgrid)
pdet_for_interpolant = np.zeros([len(grid) for grid in grids])

meshcoord = []
meshgrid = []

n_iter = 0
for i, m1 in enumerate(m1grid):
    for j, m2 in zip(range(i, len(m2grid)), m2grid[i:]):
        for k, z in enumerate(zgrid):
            n_iter += 1

with tqdm(total=n_iter, desc="Populating pdet_for_interpolant and meshgrid") as pbar:
    for i, m1 in enumerate(m1grid):
        for j, m2 in zip(range(i, len(m2grid)), m2grid[i:]):
            for k, z in enumerate(zgrid):
                already_known=False
                m12_min = m1 if m1 <= m2 else m2
                for z_max, m_min in z_max_m_min:
                    if z <= z_max and m12_min >= m_min:
                        pdet_for_interpolant[i, j, k] = 1
                        pdet_for_interpolant[j, i, k] = 1
                        already_known=True
                if not already_known:
                    m12_max = m1 if m1 >= m2 else m2
                    for z_min, m_max in z_min_m_max:
                        if z >= z_min and m12_max <= m_max:
                            already_known=True
                if not already_known:
                    meshcoord.append((i, j, k))
                    meshgrid.append((m1 ,m2, z))
                pbar.update(1)

meshcoord = np.array(meshcoord)
meshgrid = np.array(meshgrid)

# Shuffle the arrays to better ditribute load across processors
p = np.random.permutation(len(meshcoord))

meshcoord = meshcoord[p]
meshgrid = meshgrid[p]

print(np.prod(pdet_for_interpolant.shape), meshcoord.shape)

def compute_pdet_for_interpolant(meshgrid, meshcoord=None, m1grid=None, m2grid=None, zgrid=None, pdet_for_interpolant=None,
                                 n_sample=1e3, n_jobs=-1, verbose=11,
                                 from_checkpoint=False, save_checkpoint=False, checkpoint_file_path="", save_step=-1):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    logger.propagate = False
    
    if not logger.handlers:
        file_handler = logging.FileHandler("pdet_gwbench_grid_making.log")
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
    chunks_integrity_error_msg = "Ops, something went wrong while splitting meshgrid and meshcoords. Hopefully you will never see this message."
    save_step = -save_step * cpu_count() - 1 if save_step < 0 else save_step
    
    if not from_checkpoint:
        if not save_checkpoint: # Just return meshevalues
            meshvalues = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(pdet)(m1, m2, z, n_samples=n_sample, n_jobs=1)
                                                                           for m1, m2, z in meshgrid))
            return meshvalues
        else:
            # Check parameters
            assert checkpoint_file_path != "", "Provide a checkpoint file path. Extension is not needed if the checkpoint file doesn't exist yet."
            assert meshcoord is not None, "Provide also the meshcoord array. It will be saved in the checkpoint file."
            assert meshgrid.shape == meshcoord.shape, "Check the shapes of the arrays."
            pars_present = [par is not None for par in (m1grid, m2grid, zgrid, pdet_for_interpolant)]
            pars_are_present = True
            for par_is_present in pars_present:
                pars_are_present = pars_are_present and par_is_present
            assert pars_are_present, "Provide also the m1grid, m2grid, zgrid and pdet_for_interpolant arrays. They will be saved in the checkpoint file."
            assert 1 <= save_step <= len(meshgrid), f"The saving step must be greater than 1 and smaller than the length of meshgrid ({len(meshgrid)})."
            logger.info("Starting pdet computation with gwbench on the provided m1, m2, z grid using checkpoint file...")
            logger.info(f"Splitting grids into chunks with length of roughly {save_step}.")
            # Split the grid into chunks based on save_step parameter
            mgridchunks = np.array_split(meshgrid, int(len(meshgrid) / save_step))
            mcoordchunks = np.array_split(meshcoord, int(len(meshcoord) / save_step))
            # Check chunks integrity
            assert sum([len(grid) for grid in mgridchunks]) == len(meshgrid) and sum([len(grid) for grid in mcoordchunks]) == len(meshcoord), chunks_integrity_error_msg
            chunks_lengths = [len(chunk) for chunk in mgridchunks]
            logger.info(f"Grids with length of {len(meshgrid)} divided into {len(mgridchunks)} chunks. " +
                        f"{chunks_lengths.count(min(chunks_lengths))} with length {min(chunks_lengths)} and " +
                        f"{chunks_lengths.count(max(chunks_lengths))} with length {max(chunks_lengths)}")
            
            # Initialize results variables
            completed_tasks = 0
            meshvalues = np.array([])

            # Initialize checkpoint file.
            # n_sample and completed_tasks are not arrays, therefore you will need to int() the 0-dim array associated with them while reading the file
            logger.info(f"Initializing checkpoint file at {checkpoint_file_path}.")
            np.savez(checkpoint_file_path, m1grid=m1grid, m2grid=m2grid, zgrid=zgrid, pdet_for_interpolant=pdet_for_interpolant,
                     meshgrid=meshgrid, meshcoord=meshcoord, n_sample=n_sample, meshvalues=meshvalues, completed_tasks=completed_tasks)
            logger.info(f"Checkpoint file created at {checkpoint_file_path}.")
            
            # Compute and save the meshvalues for each chunk
            logger.info("Starting pdet computation...")
            for i, (mgrid, mcoord) in enumerate(zip(mgridchunks, mcoordchunks)):
                assert len(mgrid) == len(mcoord), chunks_integrity_error_msg
                logger.info(f"Working on chunk n. {i + 1} of {len(mgridchunks)}. {completed_tasks} of {len(meshgrid)} tasks completed.")
                mvalues = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(pdet)(m1, m2, z, n_samples=n_sample, n_jobs=1)
                                                                            for m1, m2, z in mgrid))
                meshvalues = np.concatenate((meshvalues, mvalues))
                completed_tasks += len(mgrid)
                np.savez(checkpoint_file_path, m1grid=m1grid, m2grid=m2grid, zgrid=zgrid, pdet_for_interpolant=pdet_for_interpolant,
                         meshgrid=meshgrid, meshcoord=meshcoord, n_sample=n_sample, meshvalues=meshvalues, completed_tasks=completed_tasks)
                logger.info(f"Chunk n. {i + 1} completed. Progress saved.")

            logger.info("Computation completed.")
            # If everything went smooth, return meshvalues
            return meshvalues
    else: # Any parameter but for n_jobs, verbose and save_step will be ignored
        assert checkpoint_file_path != "", "Provide an existing checkpoint file path."
        logger.info("Resuming pdet computation from checkpoint file...")
        # Load the checkpoint file
        logger.info(f"Loading the checkpoint file at {checkpoint_file_path}.")
        chk_file = np.load(checkpoint_file_path)
        logger.info(f"Checkpoint file at {checkpoint_file_path} loaded.")
        
        # Check parameters
        completed_tasks = int(chk_file['completed_tasks'])
        meshgrid = chk_file['meshgrid']
        if completed_tasks >= len(meshgrid):
            chk_file.close()
        assert completed_tasks < len(meshgrid), "Checkpoint file exhausted. Read meshvalues from the file"
        assert 1 <= save_step <= len(meshgrid[completed_tasks:]), f"The saving step must be greater than 1 and smaller than the length of the remaining meshgrid ({len(meshgrid[completed_tasks:])})."

        # Load the remaining variables
        n_sample = int(chk_file['n_sample'])
        meshvalues = chk_file['meshvalues']
        meshcoord = chk_file['meshcoord']
        m1grid = chk_file['m1grid']
        m2grid = chk_file['m2grid']
        zgrid = chk_file['zgrid']
        pdet_for_interpolant = chk_file['pdet_for_interpolant']
        chk_file.close()

        logger.info(f"{len(meshgrid[completed_tasks:]) - completed_tasks} tasks remaining.")

        # Split the remaining grid into chunks based on save_step parameter
        logger.info(f"Splitting remaining grids into chunks with length of roughly {save_step}.")
        mgridchunks = np.array_split(meshgrid[completed_tasks:], int(len(meshgrid[completed_tasks:]) / save_step))
        mcoordchunks = np.array_split(meshcoord[completed_tasks:], int(len(meshcoord[completed_tasks:]) / save_step))
        # Check chunks integrity
        assert sum([len(grid) for grid in mgridchunks]) == len(meshgrid[completed_tasks:]) and sum([len(grid) for grid in mcoordchunks]) == len(meshcoord[completed_tasks:]), chunks_integrity_error_msg
        chunks_lengths = [len(chunk) for chunk in mgridchunks]
        logger.info(f"Remaining grids with length of {len(meshgrid[completed_tasks:])} divided into {len(mgridchunks)} chunks. " +
                    f"{chunks_lengths.count(min(chunks_lengths))} with length {min(chunks_lengths)} and " +
                    f"{chunks_lengths.count(max(chunks_lengths))} with length {max(chunks_lengths)}")

        # Compute and save the meshvalues for each chunk
        logger.info("Resuming pdet computation...")
        for i, (mgrid, mcoord) in enumerate(zip(mgridchunks, mcoordchunks)):
            assert len(mgrid) == len(mcoord), chunks_integrity_error_msg
            logger.info(f"Working on chunk n. {i + 1} of {len(mgridchunks)}. {completed_tasks} of {len(meshgrid)} tasks completed.")
            mvalues = np.array(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(pdet)(m1, m2, z, n_samples=n_sample, n_jobs=1)
                                                                        for m1, m2, z in mgrid))
            meshvalues = np.concatenate((meshvalues, mvalues))
            completed_tasks += len(mgrid)
            np.savez(checkpoint_file_path, m1grid=m1grid, m2grid=m2grid, zgrid=zgrid, pdet_for_interpolant=pdet_for_interpolant,
                     meshgrid=meshgrid, meshcoord=meshcoord, n_sample=n_sample, meshvalues=meshvalues, completed_tasks=completed_tasks)
            logger.info(f"Chunk n. {i + 1} completed. Progress saved.")

        logger.info("Computation completed.")
        if completed_tasks != len(meshgrid):
            logger.warning(f"There is a mismatch between the length of meshgrid ({len(meshgrid)}) and the completed tasks ({completed_tasks})")
        # If everything went smooth, return meshvalues
        return meshvalues

meshvalues = compute_pdet_for_interpolant(meshgrid, meshcoord, m1grid, m2grid, zgrid, pdet_for_interpolant,
                                          n_sample=5e2,
                                          save_checkpoint=True,
                                          save_step=-10,
                                          checkpoint_file_path="data/pdet_computation_checkpoint.npz",
                                          from_checkpoint=True)

for ijk, val in zip(meshcoord, meshvalues):
    i, j, k = ijk
    pdet_for_interpolant[i, j, k] = val
    pdet_for_interpolant[j, i, k] = val

np.savez("data/pdet_nsamples_5e2_(attempt_4)", m1grid=m1grid, m2grid=m2grid, zgrid=zgrid, pdet_for_interpolant=pdet_for_interpolant)
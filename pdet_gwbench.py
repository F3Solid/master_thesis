import numpy as np
from comp_pdet import pdet
from joblib import Parallel, delayed, cpu_count
from tqdm.auto import tqdm
import logging

# Initializes meshgrid, meshcoord and pdet_for_interpolant from fresh m1, m2 and z grids
# Assumes that pdet is symmetric for mass swapping: for every (i, j, k) with i!=j, (j, i, k) is not included
# grids=(m1grid, m2grid, zgrid)
# z_max_m_min, z_min_m_max = ((z1, m1), (z2, m2), ...) are tuples defining combinations of redshifts and masses
# giving respectively pdet=1 and pdet=0. For example, for z_max_m_min, anything with lower redshift and higher minumum mass will have pdet=1;
# for z_min_m_max, anything with higher redshift and lower maximum mass will have pdet=0
# If shuffle=True meshcoord and meshgrid will be shuffled before being returned
# If pdet_for_interpolant is given then its NaN values will be initialized if possible
def initialize_pdet_grids(grids, z_max_m_min, z_min_m_max, pdet_for_interpolant=None, shuffle=True):
    def fill_pdet_matrix(grids, pdet_for_interpolant):
        # Numpy meshgrid magic for advanced indexing
        M1, M2, Z = np.meshgrid(*grids, indexing='ij')
        IJK = np.meshgrid(*[np.arange(len(grid)) for grid in grids], indexing='ij')
        for z_max, m_min in z_max_m_min:
            mass_mask = (M1 >= m_min) & (M2 >= m_min)
            z_mask = Z <= z_max
            nan_mask = np.isnan(pdet_for_interpolant)
            mask = mass_mask & z_mask & nan_mask
            pdet_for_interpolant[*[indx[mask] for indx in IJK]] = 1
        for z_min, m_max in z_min_m_max:
            mass_mask = (M1 <= m_max) & (M2 <= m_max)
            z_mask = Z >= z_min
            nan_mask = np.isnan(pdet_for_interpolant)
            mask = mass_mask & z_mask & nan_mask
            pdet_for_interpolant[*[indx[mask] for indx in IJK]] = 0

    if pdet_for_interpolant is None:
        pdet_for_interpolant = np.full([len(grid) for grid in grids], np.nan)
    else:
        shape_err_msg = f"pdet_for_interpolant has shape {pdet_for_interpolant.shape}. It must match the given grids with shape {tuple([len(grid) for grid in grids])}."
        assert pdet_for_interpolant.shape == tuple([len(grid) for grid in grids]), shape_err_msg
        pdet_for_interpolant = np.copy(pdet_for_interpolant)
        
    fill_pdet_matrix(grids, pdet_for_interpolant)
    meshcoord, meshgrid = get_meshgrid_from_pdet_grids((*grids, pdet_for_interpolant))

    # Shuffle the arrays to better ditribute load across processors
    if shuffle:
        p = np.random.permutation(len(meshcoord))
        
        meshcoord = meshcoord[p]
        meshgrid = meshgrid[p]

    print(f"pdet_for_interpolant number of points: {np.prod(pdet_for_interpolant.shape)}; meshcoord shape: {meshcoord.shape}")

    return meshcoord, meshgrid, pdet_for_interpolant

# Merges and updates two sets of (m1grid, m2grid, zgrid, pdet_for_interpolant) so that the returned set
# contains all elements from the two sets only one time and old elements are overwritten with the new ones if requested
# new_grids=(m1grid, m2grid, zgrid, pdet_for_interpolant)
# if overwrite=False existing pdet won't be changed, otherwise they will be overwritten with new values if present
def merge_pdet_grids(new_grids, existing_grid_file_path="", overwrite=False):
    # Rounding array values is critical for result reproducibility on different machines
    # due to different floating point handling within numpy routines between different CPUs
    decimals = 12 # Number of decimals to round array values to
    old_grids = np.load(existing_grid_file_path) # keys: m1grid, m2grid, zgrid, pdet_for_interpolant
    keys = ("m1grid", "m2grid", "zgrid", "pdet_for_interpolant")
    assert len(old_grids) == len(new_grids) == 4, "The new_grids list must be of length 4, containing the new m1grid, m2grid, zgrid and pdet_for_interpolant"
    new_grids = {key: val for key, val in zip(keys, new_grids)} # m1grid, m2grid, zgrid, pdet_for_interpolant

    # Make the new grids
    # keys: m1grid, m2grid, zgrid
    unq_grids = {key: np.unique(np.round(np.concatenate((old_grids[key], new_grids[key])), decimals=decimals))
                 for key in keys[:-1]}

    # Make the new pdet matrix
    pdet_for_interpolant = np.full([len(grid) for grid in unq_grids.values()], np.nan)

    grids_to_process = [old_grids, new_grids]
    for grids in grids_to_process:
        # Get indexes of input grids elements inside of the new unq_grid
        ijk = {key: np.searchsorted(unq_grids[key], np.round(grids[key], decimals=decimals))
               for key in unq_grids.keys()}
        # Numpy meshgrid magic for advanced indexing
        IJK = np.meshgrid(*ijk.values(), indexing='ij')

        # Fill pdet matrix according to the selected indexes
        if overwrite:
            pdet_for_interpolant[*IJK] = grids["pdet_for_interpolant"]
        else:
            nan_mask = np.isnan(pdet_for_interpolant[*IJK])
            pdet_for_interpolant[*[indx[nan_mask] for indx in IJK]] = grids["pdet_for_interpolant"][nan_mask]
    
    '''
    # Return only the merged grids
    for key in unq_grids.keys():
        unq_grids[key] = unq_grids[key].values
    '''
    unq_grids["pdet_for_interpolant"] = pdet_for_interpolant
    
    return unq_grids

# Get meshgrid and meshcoord for pdet computation for an already processed pdet_for_interpolant matrix
# Assumes that pdet is symmetric for mass swapping: for every (i, j, k) with i!=j, (j, i, k) is not included
# and the underlying m1, m2 and z grid
# grids=(m1grid, m2grid, zgrid, pdet_for_interpolant)
def get_meshgrid_from_pdet_grids(grids):
    # Initalize variables
    m1grid, m2grid, zgrid, pdet_for_interpolant = grids

    # select NaN elements indexes
    nan_coord = np.argwhere(np.isnan(pdet_for_interpolant))
    # select all indexes with m1=m2 or m1 < m2
    mesh_indexes = np.squeeze(np.argwhere((nan_coord[:, 0] == nan_coord[:, 1]) | (nan_coord[:, 0] < nan_coord[:, 1])))
    # Get meshcoord
    meshcoord = nan_coord[mesh_indexes]

    # Numpy meshgrid magic for getting coordinates
    M1_M2_Z = np.meshgrid(m1grid, m2grid, zgrid, indexing='ij')
    # Get meshgrid using meshcoord as a mask on the numpy meshgrid output
    # Transpose and unpackage due to shapes:
    # meshcoord: (N, 3)
    # MG: (len(m1grid), len(m2grid), len(zgrid))
    meshgrid = np.array([MG[*meshcoord.T] for MG in M1_M2_Z]).T

    return meshcoord, meshgrid

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
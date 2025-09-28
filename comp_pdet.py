from joblib import Parallel, delayed
from time import time as tt
import numpy as np
from gwbench import Network, angle_sampler, M_of_Mc_eta, f_isco_Msolar, get_cartesian_from_spherical
from GWTC4 import spin_sampler_spherical
from astropy.cosmology import Planck18
from scipy.stats import norm

np.set_printoptions(linewidth=200)

# waveform and gwbench settings
wf_model_name = "lal_bbh"
wf_other_var_dic = {"approximant": "IMRPhenomXHM"}
use_rot = True  # take the effect of Earth's rotation into account, relevant for long signals in 3G detectors
only_net = False  # calculate the SNR only for the network (True) or also for individual detectors (False)

# # fixed parameters while sampling over angles for pdet calculation
# inj_params = {
#     "Mc": 30,  # chirp mass in solar masses
#     "eta": 0.24,  # symmetric mass ratio
#     "chi1x": 0.0,  # TODO check with the others, but should be fine
#     "chi1y": 0.0,  # TODO check with the others, but should be fine
#     "chi1z": 0.5,  # TODO check with the others, but should be fine
#     "chi2x": 0.0,  # TODO check with the others, but should be fine
#     "chi2y": 0.0,  # TODO check with the others, but should be fine
#     "chi2z": 0.4,  # TODO check with the others, but should be fine
#     "DL": 500.0,  # luminosity distance in Mpc
#     "tc": 0.0,  # TODO check with the others, but should be fine
#     "phic": 0.0,  # TODO check with the others, but should be fine
# }

# # frequency array for waveform and SNR calculation
# f_lo = 1.0
# f_isco = f_isco_Msolar(M_of_Mc_eta(inj_params["Mc"], inj_params["eta"]))
# f_hi = min(4 * f_isco, 1024.0)
# df = 2.0**-4
# f = np.arange(f_lo, f_hi + df, df)

# specify the detector network
network_spec = ["CE-40_CEA", "ET-10-XYL_ETS1", "ET-10-XYL_ETS2", "ET-10-XYL_ETS3"] # CE + triangular ET
snr_thr = 8 * 2 ** 0.5

# sample angles for pdet calculation
seed = None  # fix seed for reproducibility, or None for random
# num_samples = int(1e2)  # number of samples for pdet calculation
# iotas, ras, decs, psis = angle_sampler(num_samples, seed)

def _snr(f, inj_params, iota, ra, dec, psi, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z):
    # set angles for this sample
    inj_params["iota"] = iota
    inj_params["ra"] = ra
    inj_params["dec"] = dec
    inj_params["psi"] = psi
    inj_params["chi1x"] = 0.0
    inj_params["chi1y"] = 0.0
    inj_params["chi1z"] = chi1z
    inj_params["chi2x"] = 0.0
    inj_params["chi2y"] = 0.0
    inj_params["chi2z"] = chi2z

    # initialize network object, pass variables, and calculate SNRs
    net = Network(network_spec, logger_name="Network", logger_level="WARNING")
    net.set_net_vars(wf_model_name=wf_model_name,
                     wf_other_var_dic=wf_other_var_dic,
                     use_rot=use_rot,
                     f=f,
                     inj_params=inj_params,)
    net.calc_snrs(only_net=only_net)
    # return individual detector SNRs and network SNR in last position
    return np.array([det.snr for det in net.detectors] + [net.snr])

# q = m2 / m1
def snr(m1, m2, z, n_samples=1e2, n_jobs=-1):
    if m2 < m1:
        q = m2 / m1
    else:
        q = m1 / m2
    eta = q / (1 + q) ** 2
    M = m1 + m2
    '''
    inj_params = {
        "Mc": M * eta ** (3 / 5),  # chirp mass in solar masses
        "eta": eta,  # symmetric mass ratio
        "chi1x": 0.0,  # TODO check with the others, but should be fine
        "chi1y": 0.0,  # TODO check with the others, but should be fine
        "chi1z": 0.5,  # TODO check with the others, but should be fine
        "chi2x": 0.0,  # TODO check with the others, but should be fine
        "chi2y": 0.0,  # TODO check with the others, but should be fine
        "chi2z": 0.4,  # TODO check with the others, but should be fine
        "DL": Planck18.luminosity_distance(z).to_value(),  # luminosity distance in Mpc
        "tc": 0.0,  # TODO check with the others, but should be fine
        "phic": 0.0,  # TODO check with the others, but should be fine
        }
    '''
    inj_params = {
        "Mc": M * eta ** (3 / 5),  # chirp mass in solar masses
        "eta": eta,  # symmetric mass ratio
        "DL": Planck18.luminosity_distance(z).to_value(),  # luminosity distance in Mpc
        "tc": 0.0,  # TODO check with the others, but should be fine
        "phic": 0.0,  # TODO check with the others, but should be fine
        }
    
    # frequency array for waveform and SNR calculation
    f_lo = 1.0
    f_isco = f_isco_Msolar(M)
    f_hi = min(4 * f_isco, 1024.0)
    df = 2.0**-4
    f = np.arange(f_lo, f_hi + df, df)
    
    iotas, ras, decs, psis = angle_sampler(int(n_samples), seed)
    spin1s, spin2s = spin_sampler_spherical(int(n_samples), seed)
    chi1xs, chi1ys, chi1zs = get_cartesian_from_spherical(*spin1s)
    chi2xs, chi2ys, chi2zs = get_cartesian_from_spherical(*spin2s)
    
    snrs = np.array(Parallel(n_jobs=n_jobs)(delayed(_snr)(f, inj_params, iota, ra, dec, psi, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)
                                            for iota, ra, dec, psi, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z in
                                            zip(iotas, ras, decs, psis, chi1xs, chi1ys, chi1zs, chi2xs, chi2ys, chi2zs)))

    return snrs

def pdet(m1, m2, z, n_samples=1e4, n_jobs=-1):
    snrs = snr(m1, m2, z, n_samples, n_jobs)
    n_det = 0
    for _snr in snrs[:, -1]:
        if norm.rvs(loc=_snr) > snr_thr:
            n_det += 1
    
    return n_det / int(n_samples)

# # run the sampling in parallel using joblib
# ttt = tt()
# snrs = np.array(
#     Parallel(n_jobs=4)(delayed(snr)(isamp) for isamp in range(num_samples))
# )
# print(tt() - ttt)
# print(snrs)

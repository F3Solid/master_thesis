import numpy as np
from scipy.stats import rv_histogram, truncnorm
from sklearn.neighbors import KernelDensity
from popsummary.popresult import PopulationResult

import logging
logging.getLogger('root').setLevel(logging.ERROR)

# https://zenodo.org/records/16911563
filename = "data/GWTC4/BBHSpin_MagTruncnormIidTiltIsotropicTruncnormNid.h5"
result = PopulationResult(fname=filename)

# print(result.get_metadata('hyperparameters'))

# See Eq. B26 and B27 and Fig. 17 here: https://ui.adsabs.harvard.edu/abs/2025arXiv250818083T/abstract
hyperparameters=['mu_chi', 'sigma_chi', 'mu_spin', 'sigma_spin', 'xi_spin']
par_error_msg = f"Parameter not found. Available parameters are: {', '.join(hyperparameters)}."
hyperposterior_samples = np.array(result.get_hyperparameter_samples(hyperparameters=hyperparameters)).T

hyperposteriors_marginal = dict.fromkeys(hyperparameters)
hypersamplers_marginal = dict.fromkeys(hyperparameters)
for key, vals in zip(hyperparameters, hyperposterior_samples):
    hyperposteriors_marginal[key] = vals
    hypersamplers_marginal[key] = rv_histogram(np.histogram(vals, bins='auto', density=False), density=False)

# See Eq. B26 and B27 and Fig. 17 here: https://ui.adsabs.harvard.edu/abs/2025arXiv250818083T/abstract
hypersamplers_joint = dict.fromkeys(['spin_mag', 'spin_tilt'])
model_error_msg = f"Model not found. Available models are: {', '.join(hypersamplers_joint)}."
hypersamplers_joint['spin_mag'] = KernelDensity(kernel='tophat', bandwidth=0.001).fit(hyperposterior_samples[:2].T)
hypersamplers_joint['spin_tilt'] = KernelDensity(kernel='tophat', bandwidth=0.001).fit(hyperposterior_samples[2:5].T)

# This function samples from the marginal posterior distribution of the specified parameter
# Returned shape=(size,)
def hypersampler_marginal(par, size=1, seed=None):
    assert par in hyperparameters, par_error_msg
    rng = np.random.RandomState(seed)
    return hypersamplers_marginal[par].rvs(size=size, random_state=rng)

# This function samples from the joint posterior distribution of the specified model parameter space
# Returned shape=(model_n_par, size)
def hypersampler_joint(model, size=1, seed=None):
    assert model in hypersamplers_joint, model_error_msg
    rng = np.random.RandomState(seed)
    return hypersamplers_joint[model].sample(size, rng).T

# This functions samples pairs of IID spin magnitudes from their distributions defined by the sampled hyperparameters
# Returned shape=(size, 2)
def sampler_spin_mag(size=1, seed=None):
    rng = np.random.RandomState(seed) if not isinstance(seed, np.random.mtrand.RandomState) else seed
    mu, sigma = hypersamplers_joint['spin_mag'].sample(size, rng).T
    a, b = (0 - mu) / sigma, (1 - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=(2, size), random_state=rng).T

# This functions samples pairs of NID spin cosines of tilt angles from their distributions defined by the sampled hyperparameters
# Returned shape=(size, 2)
def sampler_spin_tilt(size=1, seed=None):
    rng = np.random.RandomState(seed) if not isinstance(seed, np.random.mtrand.RandomState) else seed
    spins_tilt = np.zeros((size, 2))
    u = rng.uniform(size=size)
    mu, sigma, xi = hypersamplers_joint['spin_tilt'].sample(size, rng).T
    mixture_mask = u < xi
    num_gaus = np.sum(mixture_mask)
    num_iso = size - np.sum(mixture_mask)

    if num_gaus > 0:
        _mu, _sigma = mu[mixture_mask], sigma[mixture_mask]
        a, b = (-1 - _mu) / _sigma, (1 - _mu) / _sigma
        spins_tilt[mixture_mask] = truncnorm.rvs(a, b, loc=_mu, scale=_sigma, size=(2, num_gaus), random_state=rng).T

    if num_iso > 0:
        spins_tilt[~mixture_mask] = rng.uniform(-1, 1, size=(num_iso, 2))
        
    return spins_tilt

# Generates pairs of spins from their distribution defined by the sampled hyperparameters
def spin_sampler_spherical(size=1, seed=None):
    rng = np.random.RandomState(seed)
    
    mag = sampler_spin_mag(size, rng).T
    theta = np.arccos(sampler_spin_tilt(size, rng)).T
    phi = rng.uniform(0, 2 * np.pi, size=(size, 2)).T

    return (mag[0], theta[0], phi[0]), (mag[1], theta[1], phi[1])
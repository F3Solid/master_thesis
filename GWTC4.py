import numpy as np
from scipy.stats import rv_histogram
from popsummary.popresult import PopulationResult

# https://zenodo.org/records/16911563
filename = "data/GWTC4/BBHSpin_MagTruncnormIidTiltIsotropicTruncnormNid.h5"
result = PopulationResult(fname=filename)

# print(result.get_metadata('hyperparameters'))

# See Eq. B26 and B27 and Fig. 17 here: https://ui.adsabs.harvard.edu/abs/2025arXiv250818083T/abstract
hyperparameters=['mu_chi', 'sigma_chi', 'mu_spin', 'sigma_spin', 'xi_spin']
hyperposterior_samples = np.array(result.get_hyperparameter_samples(hyperparameters=hyperparameters)).T

hyperposteriors = dict.fromkeys(hyperparameters)
hypersamplers = dict.fromkeys(hyperparameters)
for key, vals in zip(hyperparameters, hyperposterior_samples):
    hyperposteriors[key] = vals
    hypersamplers[key] = rv_histogram(np.histogram(vals, bins='auto', density=False), density=False)

# This function samples from the posterior distribution of the specified parameter
def hypersampler(par, size=1):
    return hypersamplers[par].rvs(size=size)
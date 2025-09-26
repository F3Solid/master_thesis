import numpy as np
import scipy
from popsummary.popresult import PopulationResult

filename = "data/GWTC4/BBHSpin_MagTruncnormIidTiltIsotropicTruncnormNid.h5"
result = PopulationResult(fname=filename)

print(result.get_metadata('hyperparameters'))
print(result.get_metadata('model_names'))

# print(help(result))

hyperposterior_samples = np.array(result.get_hyperparameter_samples())
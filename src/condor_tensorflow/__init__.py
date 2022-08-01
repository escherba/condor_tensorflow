"""
Condor-TensorFlow - Condor ordinal loss regression
"""
from .version import __version__

from .loss import CondorNegLogLikelihood
from .loss import CondorOrdinalCrossEntropy
from .loss import OrdinalEarthMoversDistance
from .metrics import OrdinalMeanAbsoluteError
from .metrics import OrdinalAccuracy
from .activations import ordinal_softmax
from .labelencoder import CondorOrdinalEncoder

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
    'CondorNegLogLikelihood',
    'OrdinalAccuracy',
    'OrdinalMeanAbsoluteError',
    'OrdinalEarthMoversDistance',
    'CondorOrdinalCrossEntropy',
    'ordinal_softmax',
    'CondorOrdinalEncoder',
]

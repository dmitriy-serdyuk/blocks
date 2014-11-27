"""Annonated computation graph management."""

import logging

import theano
from theano import Variable
from theano.scalar import ScalarConstant
from theano.tensor import TensorConstant
from theano.tensor.sharedvar import SharedVariable
from theano.tensor.shared_randomstreams import RandomStreams

logger = logging.getLogger(__name__)


class Model(object):
    """Encapsulates a managed Theano computation graph.

    Attributes
    ----------
    inputs : list of Theano variables
        The inputs of the computation graph.
    outputs : list of Theano variables
        The outputs of the computations graph.

    Parameters
    ----------
    outputs : list of Theano variables
        The outputs of the computation graph.

    """
    def __init__(self, outputs):
        if isinstance(outputs, Variable):
            outputs = [outputs]
        self.outputs = outputs
        self._get_variables()

    def _get_variables(self):
        def recursion(current):
            self.variables.add(current)
            if current.owner is not None:
                owner = current.owner
                if owner not in self.applies:
                    if hasattr(owner.tag, 'updates'):
                        logger.debug("updates of {}".format(owner))
                        self.updates.extend(owner.tag.updates.items())
                    self.applies.add(owner)

                for inp in owner.inputs:
                    if inp not in self.variables:
                        recursion(inp)

        def is_input(variable):
            return (not variable.owner
                    and not isinstance(variable, SharedVariable)
                    and not isinstance(variable, TensorConstant)
                    and not isinstance(variable, ScalarConstant))

        self.variables = set()
        self.applies = set()
        self.updates = []
        for output in self.outputs:
            recursion(output)
        self.inputs = [v for v in self.variables if is_input(v)]

    def replace(self, replacements):
        """Replace certain variables in the computation graph.

        .. todo::

            Either implement more efficiently, or make the whole
            ComputationGraph immutable and return a new one here.

        """
        self.outputs = theano.clone(self.outputs, replace=replacements)
        self._get_variables()

    def function(self):
        """Create Theano function from the graph contained."""
        return theano.function(self.inputs, self.outputs,
                               updates=self.updates)


class Cost(ComputationGraph):
    """Encapsulates a cost function of a ML model.

    Parameters
    ----------
    cost : Theano ScalarVariable
        The end variable of a cost computation graph.
    seed : int
        Random seed for generation of disturbances.

    """
    def __init__(self, cost, seed=1):
        super(Cost, self).__init__([cost])
        self.random = RandomStreams(seed)

    def actual_cost(self):
        """Actual cost after changes applied."""
        return self.outputs[0]

    def apply_noise(self, variables, level):
        """Add Gaussian noise to certain variable of the cost graph.

        Parameters
        ----------
        varibles : Theano variables
            Variables to add noise to.
        level : float
            Noise level.

        """
        replace = {}
        for variable in variables:
            replace[variable] = (variable +
                                 self.random.normal(variable.shape,
                                                    std=level))
        self.replace(replace)

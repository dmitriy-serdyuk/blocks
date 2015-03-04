import numpy
from numpy.testing import assert_allclose

import theano
from theano import tensor

from blocks.bricks import Tanh, Identity
from blocks.bricks.base import application
from blocks.bricks.recurrent import SimpleRecurrent, GatedRecurrent
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.sequence_generators import (
    SequenceGenerator, LinearReadout, TrivialEmitter,
    SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant

floatX = theano.config.floatX


class TestEmitter(TrivialEmitter):
    @application
    def cost(self, readouts, outputs):
        """Compute MSE."""
        return ((readouts - outputs) ** 2).sum(axis=readouts.ndim - 1)


def test_sequence_generator():
    """Test a sequence generator with no contexts and continuous outputs.

    Such sequence generators can be used to model e.g. dynamical systems.

    """
    rng = numpy.random.RandomState(1234)

    output_dim = 1
    dim = 20
    batch_size = 30
    n_steps = 10

    transition = SimpleRecurrent(activation=Tanh(), dim=dim,
                                 weights_init=Orthogonal())
    generator = SequenceGenerator(
        LinearReadout(readout_dim=output_dim, source_names=["states"],
                      emitter=TestEmitter()),
        transition,
        weights_init=IsotropicGaussian(0.1), biases_init=Constant(0.0),
        seed=1234)
    generator.initialize()

    # Test 'cost' method
    y = tensor.tensor3('y')
    mask = tensor.matrix('mask')
    costs = generator.cost(y, mask)
    assert costs.ndim == 2
    costs_val = theano.function([y, mask], [costs])(
        rng.uniform(size=(n_steps, batch_size, output_dim)).astype(floatX),
        numpy.ones((n_steps, batch_size), dtype=floatX))[0]
    assert costs_val.shape == (n_steps, batch_size)
    assert_allclose(costs_val.sum(), 115.593, rtol=1e-5)

    # Test 'generate' method
    states, outputs, costs = [variable.eval() for variable in
                              generator.generate(
                                  states=rng.uniform(
                                      size=(batch_size, dim)).astype(floatX),
                                  iterate=True, batch_size=batch_size,
                                  n_steps=n_steps)]
    assert states.shape == (n_steps, batch_size, dim)
    assert outputs.shape == (n_steps, batch_size, output_dim)
    assert costs.shape == (n_steps, batch_size)
    assert_allclose(outputs.sum(), -0.33683, rtol=1e-5)
    assert_allclose(states.sum(), 15.7909, rtol=1e-5)
    # There is no generation cost in this case, since generation is
    # deterministic
    assert_allclose(costs.sum(), 0.0)


def test_integer_sequence_generator():
    """Test a sequence generator with integer outputs.

    Such sequence generators can be used to e.g. model language.

    """
    rng = numpy.random.RandomState(1234)

    readout_dim = 5
    feedback_dim = 3
    dim = 20
    batch_size = 30
    n_steps = 10

    transition = GatedRecurrent(activation=Tanh(), dim=dim,
                                weights_init=Orthogonal())
    generator = SequenceGenerator(
        LinearReadout(readout_dim=readout_dim, source_names=["states"],
                      emitter=SoftmaxEmitter(theano_seed=1234),
                      feedback_brick=LookupFeedback(readout_dim,
                                                    feedback_dim)),
        transition,
        weights_init=IsotropicGaussian(0.1), biases_init=Constant(0),
        seed=1234)
    generator.initialize()

    y = tensor.lmatrix('y')
    mask = tensor.matrix('mask')
    costs = generator.cost(y, mask)
    assert costs.ndim == 2
    costs_val = theano.function([y, mask], [costs])(
        rng.randint(readout_dim, size=(n_steps, batch_size)),
        numpy.ones((n_steps, batch_size), dtype=floatX))[0]
    assert costs_val.shape == (n_steps, batch_size)
    assert_allclose(costs_val.sum(), 482.827, rtol=1e-5)

    states, outputs, costs = generator.generate(
        iterate=True, batch_size=batch_size, n_steps=n_steps)
    cg = ComputationGraph(states + outputs + costs)
    states_val, outputs_val, costs_val = theano.function(
        [], [states, outputs, costs],
        updates=cg.updates)()
    assert states_val.shape == (n_steps, batch_size, dim)
    assert outputs_val.shape == (n_steps, batch_size)
    assert outputs_val.dtype == 'int64'
    assert costs_val.shape == (n_steps, batch_size)
    assert_allclose(states_val.sum(), -17.91811, rtol=1e-5)
    assert_allclose(costs_val.sum(), 482.863, rtol=1e-5)
    assert outputs_val.sum() == 630


class TestTransition(SimpleRecurrent):
    def __init__(self, attended_dim, **kwargs):
        super(TestTransition, self).__init__(**kwargs)
        self.attended_dim = attended_dim

    @application(contexts=['attended', 'attended_mask'])
    def apply(self, *args, **kwargs):
        for context in TestTransition.apply.contexts:
            kwargs.pop(context)
        return super(TestTransition, self).apply(*args, **kwargs)

    @apply.delegate
    def apply_delegate(self):
        return super(TestTransition, self).apply

    def get_dim(self, name):
        if name == 'attended':
            return self.attended_dim
        if name == 'attended_mask':
            return 0
        return super(TestTransition, self).get_dim(name)


def test_with_attention():
    """Test a sequence generator with continuous outputs and attention."""
    rng = numpy.random.RandomState(1234)

    inp_dim = 2
    inp_len = 10
    attended_dim = 3
    attended_len = 11
    batch_size = 4
    n_steps = 30

    # For values
    def rand(size):
        return rng.uniform(size=size).astype(floatX)

    # For masks
    def generate_mask(length, batch_size):
        mask = numpy.ones((length, batch_size), dtype=floatX)
        # To make it look like read data
        for i in range(batch_size):
            mask[1 + rng.randint(0, length - 1):, i] = 0.0
        return mask

    output_vals = rand((inp_len, batch_size, inp_dim))
    output_mask_vals = generate_mask(inp_len, batch_size)
    attended_vals = rand((attended_len, batch_size, attended_dim))
    attended_mask_vals = generate_mask(attended_len, batch_size)

    transition = TestTransition(
        dim=inp_dim, attended_dim=attended_dim, activation=Identity())
    attention = SequenceContentAttention(
        transition.apply.states, match_dim=inp_dim)
    generator = SequenceGenerator(
        LinearReadout(
            readout_dim=inp_dim, source_names=["states", "glimpses"],
            emitter=TestEmitter()),
        transition=transition,
        attention=attention,
        weights_init=IsotropicGaussian(0.1), biases_init=Constant(0),
        add_contexts=False, seed=1234)
    generator.initialize()

    # Test 'cost' method
    attended = tensor.tensor3("attended")
    attended_mask = tensor.matrix("attended_mask")
    outputs = tensor.tensor3('outputs')
    mask = tensor.matrix('mask')
    costs = generator.cost(outputs, mask,
                           attended=attended, attended_mask=attended_mask)
    costs_vals = costs.eval({outputs: output_vals,
                             mask: output_mask_vals,
                             attended: attended_vals,
                             attended_mask: attended_mask_vals})
    assert costs_vals.shape == (inp_len, batch_size)
    assert_allclose(costs_vals.sum(), 33.6583, rtol=1e-5)

    # Test `generate` method
    results = (
        generator.generate(n_steps=n_steps, batch_size=attended.shape[1],
                           attended=attended, attended_mask=attended_mask))
    assert len(results) == 5
    states_vals, outputs_vals, glimpses_vals, weights_vals, costs_vals = (
        theano.function([attended, attended_mask], results)
        (attended_vals, attended_mask_vals))
    assert states_vals.shape == (n_steps, batch_size, inp_dim)
    assert states_vals.shape == outputs_vals.shape
    assert glimpses_vals.shape == (n_steps, batch_size, attended_dim)
    assert weights_vals.shape == (n_steps, batch_size, attended_len)
    assert costs_vals.shape == (n_steps, batch_size)
    assert_allclose(states_vals.sum(), 23.4172, rtol=1e-5)
    # There is no generation cost in this case, since generation is
    # deterministic
    assert_allclose(costs_vals.sum(), 0.0, rtol=1e-5)
    assert_allclose(weights_vals.sum(), 120.0, rtol=1e-5)
    assert_allclose(glimpses_vals.sum(), 199.2402, rtol=1e-5)
    assert_allclose(outputs_vals.sum(), -11.6008, rtol=1e-5)

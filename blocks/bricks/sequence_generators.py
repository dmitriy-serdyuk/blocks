"""Sequence generation framework.

Recurrent networks are often used to generate/model sequences.
Examples include language modelling, machine translation, handwriting
synthesis, etc.. A typical pattern in this context is that
sequence elements are generated one often another, and every generated
element is fed back into the recurrent network state. Sometimes
also an attention mechanism is used to condition sequence generation
on some structured input like another sequence or an image.

This module provides :class:`SequenceGenerator` that builds a sequence
generating network from three main components:

* a core recurrent transition, e.g. :class:`~blocks.bricks.recurrent.LSTM`
  or :class:`~blocks.bricks.recurrent.GatedRecurrent`

* a readout component that can produce sequence elements using
  the network state and the information from the attention mechanism

* an attention mechanism (see :mod:`~blocks.bricks.attention` for
  more information)

Implementation-wise :class:`SequenceGenerator` fully relies on
:class:`BaseSequenceGenerator`. At the level of the latter an
attention is mandatory, moreover it must be a part of the recurrent
transition (see :class:`~blocks.bricks.attention.AttentionRecurrent`).
To simulate optional attention, :class:`SequenceGenerator` wraps the
pure recurrent network in :class:`FakeAttentionRecurrent`.

"""
from abc import ABCMeta, abstractmethod

from six import add_metaclass
from theano import tensor

from blocks.bricks import Initializable, Random, Bias
from blocks.bricks.base import application, Brick, lazy
from blocks.bricks.parallel import Fork, Merge
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import recurrent
from blocks.bricks.attention import (
    AbstractAttentionRecurrent, AttentionRecurrent)
from blocks.roles import add_role, COST
from blocks.utils import dict_union, dict_subset


class BaseSequenceGenerator(Initializable):
    r"""A generic sequence generator.

    This class combines two components, a readout network and an
    attention-equipped recurrent transition, into a context-dependent
    sequence generator. Third component must be also given which
    forks feedback from the readout network to obtain inputs for the
    transition.

    The class provides two methods: :meth:`generate` and :meth:`cost`. The
    former is to actually generate sequences and the latter is to compute
    the cost of generating given sequences.

    The generation algorithm description follows.

    **Definitions and notation:**

    * States :math:`s_i` of the generator are the states of the transition
      as specified in `transition.state_names`.

    * Contexts of the generator are the contexts of the
      transition as specified in `transition.context_names`.

    * Glimpses :math:`g_i` are intermediate entities computed at every
      generation step from states, contexts and the previous step glimpses.
      They are computed in the transition's `apply` method when not given
      or by explicitly calling the transition's `take_glimpses` method. The
      set of glimpses considered is specified in
      `transition.glimpse_names`.

    * Outputs :math:`y_i` are produced at every step and form the output
      sequence. A generation cost :math:`c_i` is assigned to each output.

    **Algorithm:**

    1. Initialization.

       .. math::

           y_0 = readout.initial\_outputs(contexts)\\
           s_0, g_0 = transition.initial\_states(contexts)\\
           i = 1\\

       By default all recurrent bricks from :mod:`~blocks.bricks.recurrent`
       have trainable initial states initialized with zeros. Subclass them
       or :class:`~blocks.bricks.recurrent.BaseRecurrent` directly to get
       custom initial states.

    2. New glimpses are computed:

       .. math:: g_i = transition.take\_glimpses(
           s_{i-1}, g_{i-1}, contexts)

    3. A new output is generated by the readout and its cost is
       computed:

       .. math::

            f_{i-1} = readout.feedback(y_{i-1}) \\
            r_i = readout.readout(f_{i-1}, s_{i-1}, g_i, contexts) \\
            y_i = readout.emit(r_i) \\
            c_i = readout.cost(r_i, y_i)

       Note that the *new* glimpses and the *old* states are used at this
       step. The reason for not merging all readout methods into one is
       to make an efficient implementation of :meth:`cost` possible.

    4. New states are computed and iteration is done:

       .. math::

           f_i = readout.feedback(y_i) \\
           s_i = transition.compute\_states(s_{i-1}, g_i,
                fork.apply(f_i), contexts) \\
           i = i + 1

    5. Back to step 2 if the desired sequence
       length has not been yet reached.

    | A scheme of the algorithm described above follows.

    .. image:: /_static/sequence_generator_scheme.png
            :height: 500px
            :width: 500px

    ..

    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component of the sequence generator.
    transition : instance of :class:`AbstractAttentionRecurrent`
        The transition component of the sequence generator.
    fork : :class:`.Brick`
        The brick to compute the transition's inputs from the feedback.

    See Also
    --------
    :class:`.Initializable` : for initialization parameters

    :class:`SequenceGenerator` : more user friendly interface to this\
        brick

    """
    @lazy()
    def __init__(self, readout, transition, fork, language_model=None,
                 **kwargs):
        super(BaseSequenceGenerator, self).__init__(**kwargs)
        self.readout = readout
        self.transition = transition
        self.fork = fork
        self.language_model = language_model


        self.children = [self.readout, self.fork, self.transition]
        if self.language_model:
            self.children.append(self.language_model)

    @property
    def _state_names(self):
        return self.transition.compute_states.outputs

    @property
    def _context_names(self):
        return self.transition.apply.contexts

    @property
    def _glimpse_names(self):
        return self.transition.take_glimpses.outputs

    @property
    def _lm_state_names(self):
        return self.language_model.generate.outputs

    def _push_allocation_config(self):
        # Configure readout. That involves `get_dim` requests
        # to the transition. To make sure that it answers
        # correctly we should finish its configuration first.
        self.transition.push_allocation_config()
        transition_sources = (self._state_names + self._context_names +
                              self._glimpse_names)
        self.readout.source_dims = [self.transition.get_dim(name)
                                    if name in transition_sources
                                    else self.readout.get_dim(name)
                                    for name in self.readout.source_names]

        # Configure fork. For similar reasons as outlined above,
        # first push `readout` configuration.
        self.readout.push_allocation_config()
        feedback_name, = self.readout.feedback.outputs
        self.fork.input_dim = self.readout.get_dim(feedback_name)
        self.fork.output_dims = self.transition.get_dims(
            self.fork.apply.outputs)

    @application
    def cost(self, application_call, outputs, mask=None, **kwargs):
        """Returns the average cost over the minibatch.

        The cost is computed by averaging the sum of per token costs for
        each sequence over the minibatch.

        .. warning::
            Note that, the computed cost can be problematic when batches
            consist of vastly different sequence lengths.

        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The 3(2) dimensional tensor containing output sequences.
            The axis 0 must stand for time, the axis 1 for the
            position in the batch.
        mask : :class:`~tensor.TensorVariable`
            The binary matrix identifying fake outputs.

        Returns
        -------
        cost : :class:`~tensor.Variable`
            Theano variable for cost, computed by summing over timesteps
            and then averaging over the minibatch.

        Notes
        -----
        The contexts are expected as keyword arguments.

        Adds average cost per sequence element `AUXILIARY` variable to
        the computational graph with name ``per_sequence_element``.

        """
        # Compute the sum of costs
        costs = self.cost_matrix(outputs, mask=mask, **kwargs)
        cost = tensor.mean(costs.sum(axis=0))
        add_role(cost, COST)

        # Add auxiliary variable for per sequence element cost
        application_call.add_auxiliary_variable(
            (costs.sum() / mask.sum()) if mask is not None else costs.sum(),
            name='per_sequence_element')
        return cost

    @application
    def evaluate(self, application_call, outputs, mask=None, **kwargs):
        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)
        contexts = dict_subset(kwargs, self._context_names)
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(batch_size)))

        # Run the language model
        if self.language_model:
            lm_states = self.language_model.evaluate(
                outputs=outputs, mask=mask, as_dict=True)
            lm_states = dict_subset(lm_states, ['outputs'])
        else:
            lm_states = {}

        readouts = self.readout.readout(
            feedback=feedback,
            **dict_union(lm_states, states, glimpses, contexts))
        costs = self.readout.cost(readouts, outputs)
        if mask is not None:
            costs *= mask

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)
        return ([costs] + states.values() + [outputs] + glimpses.values())

    @evaluate.property('outputs')
    def evaluate_outputs(self):
        return (['costs'] + self._state_names + ['outputs'] +
                self._glimpse_names)

    @application
    def cost_matrix(self, outputs, mask=None, **kwargs):
        """Returns generation costs for output sequences.

        See Also
        --------
        :meth:`cost` : Scalar cost.

        """
        return self.evaluate(outputs, mask=mask)[0]

    @recurrent
    def generate(self, outputs, **kwargs):
        """A sequence generation step.

        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The outputs from the previous step.

        Notes
        -----
        The contexts, previous states and glimpses are expected as keyword
        arguments.

        """
        states = dict_subset(kwargs, self._state_names)
        contexts = dict_subset(kwargs, self._context_names)
        glimpses = dict_subset(kwargs, self._glimpse_names)
        lm_states = {}
        if self.language_model:
            lm_states = dict_subset(
                kwargs, self._lm_state_names)
            lm_states = self.language_model.generate(
                outputs, as_dict=True, iterate=False, **lm_states)

        next_glimpses = self.transition.take_glimpses(
            as_dict=True,
            **dict_union(lm_states, states, glimpses, contexts))
        next_readouts = self.readout.readout(
            feedback=self.readout.feedback(outputs),
            **dict_union(states, next_glimpses, contexts))
        next_outputs = self.readout.emit(next_readouts)
        next_costs = self.readout.cost(next_readouts, next_outputs)
        next_feedback = self.readout.feedback(next_outputs)
        next_inputs = (self.fork.apply(next_feedback, as_dict=True)
                       if self.fork else {'feedback': next_feedback})
        next_states = self.transition.compute_states(
            as_list=True,
            **dict_union(next_inputs, states, next_glimpses, contexts))
        return (next_states + [next_outputs] +
                list(next_glimpses.values()) + list(lm_states.values()) +
                [next_costs])

    @generate.delegate
    def generate_delegate(self):
        return self.transition.apply

    @generate.property('states')
    def generate_states(self):
        result = self._state_names + ['outputs'] + self._glimpse_names
        if self.language_model:
            result.extend(self._lm_state_names)
        return result

    @generate.property('outputs')
    def generate_outputs(self):
        result = self._state_names + ['outputs'] + self._glimpse_names
        if self.language_model:
            result.extend(self._lm_state_names)
        result.append('costs')
        return result

    def get_dim(self, name):
        if name in (self._state_names + self._context_names +
                    self._glimpse_names):
            return self.transition.get_dim(name)
        elif name == 'outputs':
            return self.readout.get_dim(name)
        elif self.language_model and name in self._lm_state_names:
            return self.language_model.get_dim(name)
        return super(BaseSequenceGenerator, self).get_dim(name)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        state_dict = dict(
            self.transition.initial_states(
                batch_size, as_dict=True, *args, **kwargs),
            outputs=self.readout.initial_outputs(batch_size))
        if self.language_model :
            lm_initial_states = self.language_model.initial_states(
                batch_size, *args, **kwargs)
            state_dict = dict_union(state_dict, lm_initial_states)
        return [state_dict[state_name]
                for state_name in self.generate.states]

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.generate.states


@add_metaclass(ABCMeta)
class AbstractReadout(Initializable):
    """The interface for the readout component of a sequence generator.

    The readout component of a sequence generator is a bridge between
    the core recurrent network and the output sequence.

    Parameters
    ----------
    source_names : list
        A list of the source names (outputs) that are needed for the
        readout part e.g. ``['states']`` or
        ``['states', 'weighted_averages']`` or ``['states', 'feedback']``.
    readout_dim : int
        The dimension of the readout.

    Attributes
    ----------
    source_names : list
    readout_dim : int

    See Also
    --------
    :class:`BaseSequenceGenerator` : see how exactly a readout is used

    :class:`Readout` : the typically used readout brick

    """
    @lazy(allocation=['source_names', 'readout_dim'])
    def __init__(self, source_names, readout_dim, **kwargs):
        self.source_names = source_names
        self.readout_dim = readout_dim
        super(AbstractReadout, self).__init__(**kwargs)

    @abstractmethod
    def emit(self, readouts):
        """Produce outputs from readouts.

        Parameters
        ----------
        readouts : :class:`~theano.Variable`
            Readouts produced by the :meth:`readout` method of
            a `(batch_size, readout_dim)` shape.

        """
        pass

    @abstractmethod
    def cost(self, readouts, outputs):
        """Compute generation cost of outputs given readouts.

        Parameters
        ----------
        readouts : :class:`~theano.Variable`
            Readouts produced by the :meth:`readout` method
            of a `(..., readout dim)` shape.
        outputs : :class:`~theano.Variable`
            Outputs whose cost should be computed. Should have as many
            or one less dimensions compared to `readout`. If readout has
            `n` dimensions, first `n - 1` dimensions of `outputs` should
            match with those of `readouts`.

        """
        pass

    @abstractmethod
    def initial_outputs(self, batch_size):
        """Compute initial outputs for the generator's first step.

        In the notation from the :class:`BaseSequenceGenerator`
        documentation this method should compute :math:`y_0`.

        """
        pass

    @abstractmethod
    def readout(self, **kwargs):
        r"""Compute the readout vector from states, glimpses, etc.

        Parameters
        ----------
        \*\*kwargs: dict
            Contains sequence generator states, glimpses,
            contexts and feedback from the previous outputs.

        """
        pass

    @abstractmethod
    def feedback(self, outputs):
        """Feeds outputs back to be used as inputs of the transition."""
        pass


class Readout(AbstractReadout):
    r"""Readout brick with separated emitter and feedback parts.

    :class:`Readout` combines a few bits and pieces into an object
    that can be used as the readout component in
    :class:`BaseSequenceGenerator`. This includes an emitter brick,
    to which :meth:`emit`, :meth:`cost` and :meth:`initial_outputs`
    calls are delegated, a feedback brick to which :meth:`feedback`
    functionality is delegated, and a pipeline to actually compute
    readouts from all the sources (see the `source_names` attribute
    of :class:`AbstractReadout`).

    The readout computation pipeline is constructed from `merge` and
    `post_merge` brick, whose responsibilites are described in the
    respective docstrings.

    Parameters
    ----------
    emitter : an instance of :class:`AbstractEmitter`
        The emitter component.
    feedback_brick : an instance of :class:`AbstractFeedback`
        The feedback component.
    merge : :class:`.Brick`, optional
        A brick that takes the sources given in `source_names` as an input
        and combines them into a single output. If given, `merge_prototype`
        cannot be given.
    merge_prototype : :class:`.FeedForward`, optional
        If `merge` isn't given, the transformation given by
        `merge_prototype` is applied to each input before being summed. By
        default a :class:`.Linear` transformation without biases is used.
        If given, `merge` cannot be given.
    post_merge : :class:`.Feedforward`, optional
        This transformation is applied to the merged inputs. By default
        :class:`.Bias` is used.
    merged_dim : int, optional
        The input dimension of `post_merge` i.e. the output dimension of
        `merge` (or `merge_prototype`). If not give, it is assumed to be
        the same as `readout_dim` (i.e. `post_merge` is assumed to not
        change dimensions).
    \*\*kwargs : dict
        Passed to the parent's constructor.

    See Also
    --------
    :class:`BaseSequenceGenerator` : see how exactly a readout is used

    :class:`AbstractEmitter`, :class:`AbstractFeedback`

    """
    def __init__(self, emitter=None, feedback_brick=None,
                 merge=None, merge_prototype=None,
                 post_merge=None, merged_dim=None, **kwargs):
        super(Readout, self).__init__(**kwargs)

        if not emitter:
            emitter = TrivialEmitter(self.readout_dim)
        if not feedback_brick:
            feedback_brick = TrivialFeedback(self.readout_dim)
        if not merge:
            merge = Merge(input_names=self.source_names,
                          prototype=merge_prototype)
        if not post_merge:
            post_merge = Bias(dim=self.readout_dim)
        if not merged_dim:
            merged_dim = self.readout_dim
        self.emitter = emitter
        self.feedback_brick = feedback_brick
        self.merge = merge
        self.post_merge = post_merge
        self.merged_dim = merged_dim

        self.children = [self.emitter, self.feedback_brick,
                         self.merge, self.post_merge]

    def _push_allocation_config(self):
        self.emitter.readout_dim = self.get_dim('readouts')
        self.feedback_brick.output_dim = self.get_dim('outputs')
        self.merge.input_names = self.source_names
        self.merge.input_dims = self.source_dims
        self.merge.output_dim = self.merged_dim
        self.post_merge.input_dim = self.merged_dim
        self.post_merge.output_dim = self.readout_dim

    @application
    def readout(self, **kwargs):
        merged = self.merge.apply(**{name: kwargs[name]
                                     for name in self.merge.input_names})
        merged = self.post_merge.apply(merged)
        return merged

    @application
    def emit(self, readouts):
        return self.emitter.emit(readouts)

    @application
    def cost(self, readouts, outputs):
        return self.emitter.cost(readouts, outputs)

    @application
    def initial_outputs(self, batch_size):
        return self.emitter.initial_outputs(batch_size)

    @application(outputs=['feedback'])
    def feedback(self, outputs):
        return self.feedback_brick.feedback(outputs)

    def get_dim(self, name):
        if name == 'outputs':
            return self.emitter.get_dim(name)
        elif name == 'feedback':
            return self.feedback_brick.get_dim(name)
        elif name == 'readouts':
            return self.readout_dim
        return super(Readout, self).get_dim(name)


@add_metaclass(ABCMeta)
class AbstractEmitter(Brick):
    """The interface for the emitter component of a readout.

    Attributes
    ----------
    readout_dim : int
        The dimension of the readout. Is given by the
        :class:`Readout` brick when allocation configuration
        is pushed.

    See Also
    --------
    :class:`Readout`

    :class:`SoftmaxEmitter` : for integer outputs

    """
    @abstractmethod
    def emit(self, readouts):
        """Implements the respective method of :class:`Readout`."""
        pass

    @abstractmethod
    def cost(self, readouts, outputs):
        """Implements the respective method of :class:`Readout`."""
        pass

    @abstractmethod
    def initial_outputs(self, batch_size):
        """Implements the respective method of :class:`Readout`."""
        pass


@add_metaclass(ABCMeta)
class AbstractFeedback(Brick):
    """The interface for the feedback component of a readout.

    See Also
    --------
    :class:`Readout`

    :class:`LookupFeedback` for integer outputs

    """
    @abstractmethod
    def feedback(self, outputs):
        """Implements the respective method of :class:`Readout`."""
        pass


class TrivialEmitter(AbstractEmitter):
    """An emitter for the trivial case when readouts are outputs.

    Parameters
    ----------
    readout_dim : int
        The dimension of the readout.

    Notes
    -----
    By default :meth:`cost` always returns zero tensor.

    """
    @lazy(allocation=['readout_dim'])
    def __init__(self, readout_dim, **kwargs):
        super(TrivialEmitter, self).__init__(**kwargs)
        self.readout_dim = readout_dim

    @application
    def emit(self, readouts):
        return readouts

    @application
    def cost(self, readouts, outputs):
        return tensor.zeros_like(outputs)

    @application
    def initial_outputs(self, batch_size):
        return tensor.zeros((batch_size, self.readout_dim))

    def get_dim(self, name):
        if name == 'outputs':
            return self.readout_dim
        return super(TrivialEmitter, self).get_dim(name)


class SoftmaxEmitter(AbstractEmitter, Initializable, Random):
    """A softmax emitter for the case of integer outputs.

    Interprets readout elements as energies corresponding to their indices.

    Parameters
    ----------
    initial_output : int or a scalar :class:`~theano.Variable`
        The initial output.

    """
    def __init__(self, initial_output=0, **kwargs):
        self.initial_output = initial_output
        super(SoftmaxEmitter, self).__init__(**kwargs)

    @application
    def probs(self, readouts):
        shape = readouts.shape
        return tensor.nnet.softmax(readouts.reshape(
            (tensor.prod(shape[:-1]), shape[-1]))).reshape(shape)

    @application
    def emit(self, readouts):
        probs = self.probs(readouts)
        batch_size = probs.shape[0]
        pvals_flat = probs.reshape((batch_size, -1))
        generated = self.theano_rng.multinomial(pvals=pvals_flat)
        return generated.reshape(probs.shape).argmax(axis=-1)

    @application
    def cost(self, readouts, outputs):
        # WARNING: unfortunately this application method works
        # just fine when `readouts` and `outputs` have
        # different dimensions. Be careful!
        probs = self.probs(readouts)
        max_output = probs.shape[-1]
        flat_outputs = outputs.flatten()
        num_outputs = flat_outputs.shape[0]
        return -tensor.log(
            probs.flatten()[max_output * tensor.arange(num_outputs) +
                            flat_outputs].reshape(outputs.shape))

    @application
    def initial_outputs(self, batch_size):
        return self.initial_output * tensor.ones((batch_size,), dtype='int64')

    def get_dim(self, name):
        if name == 'outputs':
            return 0
        return super(SoftmaxEmitter, self).get_dim(name)


class TrivialFeedback(AbstractFeedback):
    """A feedback brick for the case when readout are outputs."""
    @lazy(allocation=['output_dim'])
    def __init__(self, output_dim, **kwargs):
        super(TrivialFeedback, self).__init__(**kwargs)
        self.output_dim = output_dim

    @application(outputs=['feedback'])
    def feedback(self, outputs):
        return outputs

    def get_dim(self, name):
        if name == 'feedback':
            return self.output_dim
        return super(TrivialFeedback, self).get_dim(name)


class LookupFeedback(AbstractFeedback, Initializable):
    """A feedback brick for the case when readout are integers.

    Stores and retrieves distributed representations of integers.

    """
    def __init__(self, num_outputs=None, feedback_dim=None, **kwargs):
        super(LookupFeedback, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.feedback_dim = feedback_dim

        self.lookup = LookupTable(num_outputs, feedback_dim,
                                  weights_init=self.weights_init)
        self.children = [self.lookup]

    def _push_allocation_config(self):
        self.lookup.length = self.num_outputs
        self.lookup.dim = self.feedback_dim

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0
        return self.lookup.apply(outputs)

    def get_dim(self, name):
        if name == 'feedback':
            return self.feedback_dim
        return super(LookupFeedback, self).get_dim(name)


class FakeAttentionRecurrent(AbstractAttentionRecurrent, Initializable):
    """Adds fake attention interface to a transition.

    :class:`BaseSequenceGenerator` requires its transition brick to support
    :class:`~blocks.bricks.attention.AbstractAttentionRecurrent` interface,
    that is to have an embedded attention mechanism.  For the cases when no
    attention is required (e.g.  language modeling or encoder-decoder
    models), :class:`FakeAttentionRecurrent` is used to wrap a usual
    recurrent brick. The resulting brick has no glimpses and simply
    passes all states and contexts to the wrapped one.

    .. todo::

        Get rid of this brick and support attention-less transitions
        in :class:`BaseSequenceGenerator`.

    """
    def __init__(self, transition, **kwargs):
        super(FakeAttentionRecurrent, self).__init__(**kwargs)
        self.transition = transition

        self.state_names = transition.apply.states
        self.context_names = transition.apply.contexts
        self.glimpse_names = []

        self.children = [self.transition]

    @application
    def apply(self, *args, **kwargs):
        return self.transition.apply(*args, **kwargs)

    @apply.delegate
    def apply_delegate(self):
        return self.transition.apply

    @application
    def compute_states(self, *args, **kwargs):
        return self.transition.apply(iterate=False, *args, **kwargs)

    @compute_states.delegate
    def compute_states_delegate(self):
        return self.transition.apply

    @application(outputs=[])
    def take_glimpses(self, *args, **kwargs):
        return None

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        return self.transition.initial_states(batch_size,
                                              *args, **kwargs)

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.transition.apply.states

    def get_dim(self, name):
        return self.transition.get_dim(name)


class SequenceGenerator(BaseSequenceGenerator):
    r"""A more user-friendly interface for :class:`BaseSequenceGenerator`.

    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component for the sequence generator.
    transition : instance of :class:`.BaseRecurrent`
        The recurrent transition to be used in the sequence generator.
        Will be combined with `attention`, if that one is given.
    attention : object, optional
        The attention mechanism to be added to ``transition``,
        an instance of
        :class:`~blocks.bricks.attention.AbstractAttention`.
    add_contexts : bool
        If ``True``, the
        :class:`.AttentionRecurrent` wrapping the
        `transition` will add additional contexts for the attended and its
        mask.
    \*\*kwargs : dict
        All keywords arguments are passed to the base class. If `fork`
        keyword argument is not provided, :class:`.Fork` is created
        that forks all transition sequential inputs without a "mask"
        substring in them.

    """
    def __init__(self, readout, transition, attention=None,
                 add_contexts=True, **kwargs):
        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))
        if attention:
            transition = AttentionRecurrent(
                transition, attention,
                add_contexts=add_contexts, name="att_trans")
        else:
            transition = FakeAttentionRecurrent(transition,
                                                name="with_fake_attention")
        super(SequenceGenerator, self).__init__(
            readout, transition, **kwargs)

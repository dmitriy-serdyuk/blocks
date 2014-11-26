from collections import OrderedDict

from theano import Variable

from block.utils import update_instance


class MonitorChannel(object):
    """A channel to monitor

    Parameters
    ----------
    name : str
        The name of this monitor channel, should be unique in order to
        distinguish channels.
    value : Theano variable or callable
        Either a Theano variable, or a callable function, which takes a
        model and dataset as arguments.
    validation : bool, optional
        If ``True``, this channel will be part of the validation/test set
        monitoring (i.e. be calculated after a given interval). If not,
        this channel will be monitored at each update.  Defaults to
        ``True``.
    needs_data : bool, optional
        If ``True``, it means that this channel's value depends on the
        input to the model (e.g. for a cost this should be ``True``, for
        weight norms it should be ``False``). In this case, the value will
        be averaged over a series of batches. If not, the value will only
        be requested once. Is ``True`` by default. The value is ignored
        when :param:`validation` is ``False``.
    inputs : dict of str, Theano variable pairs, optional
        If the channel requires inputs which are not part of the Theano
        graph, pass them here.

    Notes
    -----
    The values returned by callable monitor channels can be any Python
    object.

    """
    def __init__(self, name, value, validation=True, needs_data=True,
                 inputs=None):
        assert validation or needs_data
        if isinstance(value, Variable):
            self.is_theano_var = True
        else:
            assert callable(value)
            self.is_theano_var = False
        if inputs is None:
            inputs = OrderedDict()
        else:
            assert self.is_theano_var
            assert isinstance(inputs, OrderedDict)
        update_instance(self, locals())

    def __call__(self, model, dataset):
        assert not self.is_theano_var
        return self.value(model, dataset)


class Monitor(object):
    """A class for monitoring Models while they are being trained.

    Records a variety of things (the objective function, reconstruction
    errors, weight norms, gradients, etc.)

    There are two kinds of monitoring. The first one calculates statistics
    over a set of monitoring datasets (usually your validation and test
    sets) after a fixed interval of updates. For small datasets/fast
    models, it often suffices to set the interval equal to the size of the
    dataset, calculating the statistics after each epoch. For larger
    datasets you might want to set two intervals: one equal to the size of
    the dataset, and one smaller interval for intermediate results.

    The second type of monitoring happens at each update. These monitoring
    channels are compiled together with the update function of the training
    algorithm. This can be useful to calculate e.g. the likelihood of your
    training batches for models where calculating statistics on a
    validation set would slow training down too much.

    Parameters
    ----------
    intervals : list of integers
        Perform monitoring on the validation sets at the given intervals.

    Attributes
    ----------
    num_examples_seen : int
        The number of examples seen by the model.

    """
    def __init__(self, intervals, dataset):
        self.num_examples_seen = -1
        self.channels = []

    def get_data_dependent_theano_vars(self):
        """Get the Theano monitoring variables that require data.

        Returns
        -------
        channels : list of Theano variables
            List of channels to monitor
        inputs : dict of str, Theano variable pairs
            Dictionary of inputs to the computional graph required to
            compute channels with their names.

        """
        channels = [channel for channel in self.channels
                    if channel.needs_data and channel.is_theano_var]
        inputs = OrderedDict()
        for channel in channels:
            inputs.update(channel.inputs)
        return channels, inputs

    def get_update_theano_vars(self):
        """Return the arguments of Theano monitor variables.

        This returns the arguments and keyword arguments needed to compile
        a Theano function that returns values to be monitored after each
        update. These values can be combined with the updates to the model
        to compile a function that both monitors and updates the model
        using a single computational graph, which is more computationally
        efficient.

        Notes
        -----
        The numerical values of these monitoring channels are expected to
        be provided as arguments (in the form of a dictionary) when calling
        the :meth:`__call__` method.

        """
        channels = [channel for channel in self.channels
                    if channel.validation and channel.is_theano_var]
        inputs = OrderedDict()
        for channel in channels:
            inputs.update(channel.inputs)
        return channels, inputs

    def __call__(self, update_monitoring_values=None):
        """Perform monitoring.

        Parameters
        ----------
        update_monitoring_values : dict of Theano variables, object pairs
            A dictionary with the channel values (Theano variables) as
            keys, and the numerical values as values.

        """
        self.num_examples_seen += 1
        if update_monitoring_values is None:
            update_monitoring_values is {}
        # Perform update monitoring
        if any(self.num_samples_seen % interval == 0
               for interval in self.intervals):
            pass  # Perform validation monitoring

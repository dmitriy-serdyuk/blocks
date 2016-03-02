import numpy
from mimir import Logger
from mimir.logger import _Logger
from mimir.serialization import serialize_numpy

from .log import TrainingLogBase


def pretty_serialize_numpy(obj):
    if isinstance(obj, numpy.ndarray):
        try:
            return numpy.asscalar(obj)
        except TypeError:
            pass
    return serialize_numpy(obj)


class PicklableLogger(_Logger):
    """A picklable wrapper around mimir logger.

    This class is a picklable version of `:class:mimir.Logger` with
    some additional functionality. The current status is saved in
    `iteration_status` attribute and is flushed to the file only if
    the next iteration is requested or the class is pickled.

    """
    def __init__(self, filename, **kwargs):
        logger = Logger(filename, default=pretty_serialize_numpy, **kwargs)
        self.__dict__.update(logger.__dict__)
        self.all_kwargs = kwargs
        self.all_kwargs['filename'] = filename
        self.status = {}
        self.iteration_status = {}
        self.last_flushed = -1

    def __setstate__(self, state):
        logger = Logger(**state)
        self.__dict__.update(logger.__dict__)

    def __getstate__(self):
        self.flush()
        return self.all_kwargs

    def flush(self):
        iterations_done = self.status['iterations_done']
        self.log({iterations_done: self.iteration_status})
        self.iteration_status = {}
        self.last_flushed = iterations_done

    def get_record(self, time):
        iterations_done = self.status.get('iterations_done', -1)
        if time > self.last_flushed:
            self.flush()
            return self.iteration_status
        elif time < iterations_done - 1:
            raise ValueError('cannot get past log entries for JSON log')
        elif time == iterations_done:
            return self.iteration_status
        else:
            return self[iterations_done - time]


class JSONLog(TrainingLogBase):
    def __init__(self, filename='test.jsonl.gz'):
        self.logger = PicklableLogger(maxlen=2, filename=filename)
        TrainingLogBase.__init__(self)

    @property
    def status(self):
        return self.logger.status

    def __getitem__(self, time):
        self._check_time(time)
        return self.logger.get_record(time)

    def __setitem__(self, time, value):
        raise ValueError('cannot manually change log')

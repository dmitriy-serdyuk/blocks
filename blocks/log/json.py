from mimir import Logger
from mimir.logger import _Logger

from .log import TrainingLogBase


class PicklableLogger(_Logger):
    """A picklable wrapper around mimir logger.

    This class is a picklable version of `:class:mimir.Logger`.

    """
    def __init__(self, **kwargs):
        self.logger_kwargs = kwargs
        self.open()

    def open(self):
        logger = Logger(**self.logger_kwargs)
        self.__dict__.update(logger.__dict__)
        self.load(self.logger_kwargs['filename'])

    def __setstate__(self, state):
        logger = Logger(**state)
        logger.load(state['filename'])
        self.logger_kwargs = state
        self.__dict__.update(logger.__dict__)

    def __getstate__(self):
        return self.logger_kwargs


class JSONLinesLog(TrainingLogBase):
    """JSON Lines log.

    The current status is saved in
    `iteration_status` attribute and is flushed to the file only if
    the next iteration is requested.

    """
    def __init__(self, filename='log.jsonl.gz', **kwargs):
        self.status = {}
        TrainingLogBase.__init__(self)
        kwargs.setdefault("maxlen", 2)
        kwargs.setdefault("filename", filename)
        kwargs.setdefault("formatter", None)
        self.logger = PicklableLogger(**kwargs)
        self.last_flushed = -1
        self.iteration_status = {}

    def flush(self):
        iterations_done = self.status['iterations_done']
        self.logger.log({iterations_done: self.iteration_status})
        self.iteration_status = {}
        self.last_flushed = iterations_done

    def __getitem__(self, time):
        self._check_time(time)
        iterations_done = self.status.get('iterations_done', -1)
        if time > self.last_flushed:
            self.flush()
            return self.iteration_status
        elif time < iterations_done - 1:
            try:
                return self.logger[iterations_done - time]
            except IndexError:
                raise ValueError(
                    'cannot get past log entries for JSON log, max log length '
                    'in memory is: {}'.format(self.logger_kwargs['maxlen']))
        elif time == iterations_done:
            return self.iteration_status
        else:
            return self.logger[iterations_done - time]

    def __setitem__(self, time, value):
        raise ValueError('cannot manually change log')

    def __enter__(self):
        self.logger.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.close()

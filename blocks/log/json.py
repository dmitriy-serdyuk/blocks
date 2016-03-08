import os.path
from mimir import Logger
from mimir.logger import _Logger

from .log import TrainingLogBase


class PicklableLogger(_Logger):
    """A picklable wrapper around mimir logger.

    This class is a picklable version of `:class:mimir.Logger`.

    """
    def __init__(self, **kwargs):
        self.logger_kwargs = kwargs
        self.opened = False

    def open(self):
        if not self.opened:
            logger = Logger(**self.logger_kwargs)
            self.__dict__.update(logger.__dict__)
            self.load(self.logger_kwargs['filename'])
            self.opened = True

    def __setstate__(self, state):
        self.logger_kwargs = state
        self.opened = False
        self.open()

    def __getstate__(self):
        return self.logger_kwargs


class JSONLinesLog(TrainingLogBase):
    """A log stored in gzipped JSON Lines format.

    Each line of the log is a dictionary of a form
    `{<iteration>: {<record_name>: <recodr_value>...}}`.

    Examples
    --------

    Analysis of the log can be easily done with
    `jq <https://stedolan.github.io/jq/>`__

    .. code:: bash
        gunzip -c log.jsonl.gz | jq '.reports.train_error'

        # Or equivalently
        zcat log.jsonl.gz | jq '.reports.train_error'

        # To filter out null entires
        zcat log.jsnol.gz | jq '.reports.train_error | select(.>0)'

        # To extract minimal training error
        gunzip -c log.jsonl.gz | jq -s '. | map(.reports.true_cost) | min'

        # To include the iteration with minimal training error
        gunzip -c log.jsonl.gz | jq -s '. |
            map([.iterations_done, .reports.true_cost]) | min_by(.[1])'

    """
    def __init__(self, filename='log.jsonl.gz', maxlen=21, formatter=None,
                 **kwargs):
        self.status = {}
        TrainingLogBase.__init__(self)
        if os.path.isfile(filename):
            os.remove(filename)
        self.logger = PicklableLogger(
            filename=filename, maxlen=maxlen, formatter=formatter, **kwargs)
        self.last_flushed = None
        self.current_row_container = {}

    def flush(self):
        iterations_done = self.status['iterations_done']
        self.logger.log({'iterations_done': iterations_done - 1,
                         'reports': self.current_row_container})
        self.current_row_container = {}
        self.last_flushed = iterations_done

    def __getitem__(self, time):
        self._check_time(time)
        iterations_done = self.status.get('iterations_done', -1)
        if self.last_flushed is None:
            self.last_flushed = time
        if time > self.last_flushed:
            self.flush()
            return self.current_row_container
        elif time < iterations_done - 1:
            try:
                return self.logger[time - 1]['reports']
            except IndexError:
                raise ValueError(
                    'cannot get past log entries for JSON log, max log length '
                    'in memory is: {}'.format(
                        self.logger.logger_kwargs['maxlen']))
        elif time == iterations_done:
            return self.current_row_container
        else:
            return self.logger[time - 1]['reports']

    def __len__(self):
        # One more because of the current row which is not yet flushed
        return len(self.logger) + 1

    def __setitem__(self, time, value):
        raise ValueError('cannot manually change JSON Lines log')

    def __enter__(self):
        self.logger.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.close()

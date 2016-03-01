from mimir import Logger

from .log import TrainingLogBase


class PiclableLogger(object):
    def __init__(self, filename, **kwargs):
        logger = Logger(**kwargs)
        self.__dict__ = logger.__dict__
        self.all_kwargs = kwargs
        self.all_kwargs['filename'] = filename
        self.status = {}
        self.iteration_status = {}

    def __setstate__(self, state):
        logger = Logger(**kwargs)
        self.__dict__ = logger.__dict__

    def __getstate__(self):
        self.flush()
        return self.all_kwargs

    def flush(self):
        self.log({self.status['iterations_done']: self.iteration_status})
        self.iteration_status = {}

    def get_record(self, time):
        iterations_done = self.status.get('iterations_done', -1)
        if time > iterations_done:
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
        self.logger = PiclableLogger(maxlen=2, filename=filename)
        TrainingLogBase.__init__(self)

    @property
    def status(self):
        return self.logger.status

    def __getitem__(self, time):
        self._check_time(time)
        return self.logger.get_record(time)

    def __setitem__(self, time, value):
        raise ValueError('cannot manually change log')

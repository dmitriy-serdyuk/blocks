from .log import TrainingLog
from .sqlite import SQLiteLog
from .json import JSONLog

BACKENDS = {
    'python': TrainingLog,
    'sqlite': SQLiteLog,
    'mimir': JSONLog
}

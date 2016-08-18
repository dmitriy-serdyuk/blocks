from numpy.testing import assert_allclose, assert_raises

from blocks.extensions import Timing, FinishAfter
from blocks.utils.testing import MockMainLoop


def test_timing():
    epochs = 2
    main_loop = MockMainLoop(delay_time=0.1,
                             extensions=[Timing(prefix='each'),
                                         Timing(prefix='each_second',
                                                every_n_epochs=2),
                                         FinishAfter(after_n_epochs=epochs)])
    main_loop.run()
    iterations = int(main_loop.log.status['iterations_done'] / epochs)
    assert_allclose(
        (main_loop.log[iterations]['each_time_train_this_epoch'] +
         main_loop.log[iterations]['each_time_train_this_epoch']) / 2,
        main_loop.log.current_row['each_second_time_train_this_epoch'],
        atol=1e-2)
    assert 'each_time_read_data_this_epoch' in main_loop.log[iterations]
    assert 'each_second_time_read_data_this_epoch' in main_loop.log[iterations]


def test_timing_rises():
    epochs = 2
    main_loop = MockMainLoop(delay_time=0.1,
                             extensions=[Timing(before_training=True),
                                         FinishAfter(after_n_epochs=epochs)])
    assert_raises(ValueError, main_loop.run)


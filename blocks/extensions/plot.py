import logging
import signal
import time
from subprocess import Popen, PIPE

try:
    from bokeh.plotting import figure, output_server, show, cursession, push
    bokeh_available = True
except ImportError:
    bokeh_available = False

from blocks.extensions import SimpleExtension

logger = logging.getLogger(__name__)


class Plot(SimpleExtension):
    """Live plotting of monitoring channels.

    In most cases it is preferable to start the Bokeh plotting server
    manually, so that your plots are stored permanently.

    Alternatively, you can set the ``start_server`` argument of this
    extension to ``True``, to automatically start a server when training
    starts. However, in that case your plots will be deleted when you shut
    down the plotting server!

    .. warning::

       When starting the server automatically using the ``start_server``
       argument, the extension won't attempt to shut down the server at the
       end of training (to make sure that you do not lose your plots the
       moment training completes). You have to shut it down manually (the
       PID will be shown in the logs). If you don't do this, this extension
       will crash when you try and train another model with
       ``start_server`` set to ``True``, because it can't run two servers
       at the same time.

    Parameters
    ----------
    document : str
        The name of the Bokeh document. Use a different name for each
        experiment if you are storing your plots.
    channels : list of lists of strings
        The names of the monitor channels that you want to plot. The
        channels in a single sublist will be plotted together in a single
        figure, so use e.g. ``[['test_cost', 'train_cost'],
        ['weight_norms']]`` to plot a single figure with the training and
        test cost, and a second figure for the weight norms.
    open_browser : bool, optional
        Whether to try and open the plotting server in a browser window.
        Defaults to ``True``. Should probably be set to ``False`` when
        running experiments non-locally (e.g. on a cluster or through SSH).
    start_server : bool, optional
        Whether to try and start the Bokeh plotting server. Defaults to
        ``False``. The server started is not persistent i.e. after shutting
        it down you will lose your plots. If you want to store your plots,
        start the server manually using the ``bokeh-server`` command. Also
        see the warning above.

    """
    # Tableau 10 colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, document, channels, open_browser=False,
                 start_server=False, **kwargs):
        if not bokeh_available:
            raise ImportError
        self.plots = {}
        self.start_server = start_server
        self._startserver()
        output_server(document)

        # Create figures for each group of channels
        self.p = []
        self.p_indices = {}
        for i, channel_set in enumerate(channels):
            self.p.append(figure(title='{} #{}'.format(document, i + 1)))
            for channel in channel_set:
                self.p_indices[channel] = i
        if open_browser:
            show()

        kwargs.setdefault('after_every_epoch', True)
        kwargs.setdefault("before_first_epoch", True)
        super(Plot, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        log = self.main_loop.log
        iteration = log.status.iterations_done
        i = 0
        for key, value in log.current_row:
            if key in self.p_indices:
                if key not in self.plots:
                    fig = self.p[self.p_indices[key]]
                    fig.line([iteration], [value], legend=key,
                             x_axis_label='iterations',
                             y_axis_label='value', name=key,
                             line_color=self.colors[i % len(self.colors)])
                    i += 1
                    renderer = fig.select(dict(name=key))
                    self.plots[key] = renderer[0].data_source
                else:
                    self.plots[key].data['x'].append(iteration)
                    self.plots[key].data['y'].append(value)
                    cursession().store_objects(self.plots[key])
        push()

    def _startserver(self):
        if self.start_server:
            def preexec_fn():
                """Prevents the server from dying on training interrupt."""
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Only memory works with subprocess, need to wait for it to start
            logger.info('Starting plotting server on localhost:5006')
            self.sub = Popen('bokeh-server --ip 0.0.0.0 '
                             '--backend memory'.split(),
                             stdout=PIPE, stderr=PIPE, preexec_fn=preexec_fn)
            time.sleep(2)
            logger.info('Plotting server PID: {}'.format(self.sub.pid))
        else:
            self.sub = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['sub'] = None
        return state

    def __setstate(self, state):
        self.__dict__.update(state)
        self._startserver()

Welcome to Blocks' documentation!
=================================
Blocks is a framework that helps you build and manage neural network models on
using Theano.

Want to get try it out? Start by :doc:`installing <setup>` Blocks and having a
look at the :ref:`quickstart <quickstart>` further down this page. Once you're
hooked, try your hand at the :ref:`tutorials <tutorials>`.

.. _tutorials:

Tutorials
---------
.. toctree::
   :maxdepth: 1

   setup
   tutorial
   bricks_overview
   cg
   plotting

In-depth
--------
.. toctree::
   :maxdepth: 1

   configuration
   datasets
   serialization
   api/index.rst
   development/index.rst

.. warning::
   Blocks is a new project which is still under development. As such, certain
   (all) parts of the framework are subject to change.

   That said, if you are interested in using Blocks and run into any problems,
   don't hesitate to file bug reports, feature requests, or simply ask for help,
   by `making a GitHub issue`_.

.. _making a GitHub issue: https://github.com/bartvm/blocks/issues/new

.. _quickstart:

Quickstart
==========

.. doctest::
   :hide:

   >>> from theano import tensor
   >>> from blocks.algorithms import GradientDescent, SteepestDescent
   >>> from blocks.bricks import MLP, Tanh, Softmax
   >>> from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
   >>> from blocks.initialization import IsotropicGaussian, Constant
   >>> from blocks.datasets.streams import DataStream
   >>> from blocks.datasets.mnist import MNIST
   >>> from blocks.datasets.schemes import SequentialScheme
   >>> from blocks.extensions import FinishAfter, Printing
   >>> from blocks.extensions.monitoring import DataStreamMonitoring
   >>> from blocks.main_loop import MainLoop

Construct your model.

>>> mlp = MLP(activations=[Tanh(), Softmax()], dims=[784, 100, 10],
...           weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
>>> mlp.initialize()

Calculate your loss function.

>>> x = tensor.matrix('features')
>>> y = tensor.lmatrix('targets')
>>> y_hat = mlp.apply(x)
>>> cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)
>>> error_rate = MisclassificationRate().apply(y.flatten(), y_hat)

Load your training data.

>>> mnist_train = MNIST("train")
>>> train_stream = DataStream(
...     dataset=mnist_train,
...     iteration_scheme=SequentialScheme(mnist_train.num_examples, 128))
>>> mnist_test = MNIST("test")
>>> test_stream = DataStream(
...     dataset=mnist_test,
...     iteration_scheme=SequentialScheme(mnist_train.num_examples, 1024))

And train!

>>> main_loop = MainLoop(
...     model=mlp, data_stream=train_stream,
...     algorithm=GradientDescent(
...         cost=cost, step_rule=SteepestDescent(learning_rate=0.1)),
...     extensions=[FinishAfter(after_n_epochs=5),
...                 DataStreamMonitoring(
...                     variables=[cost, error_rate],
...                     data_stream=test_stream,
...                     prefix="test"),
...                 Printing()])
>>> main_loop.run() # doctest: +SKIP

Features
--------

Currently Blocks supports and provides:

* Constructing parametrized Theano operations, called "bricks"
* Pattern matching to select variables and bricks in large models
* A pipeline for loading and iterating over training data
* Algorithms to optimize your model
* Saving and resuming of training
* Monitoring and analyzing values during training progress (on the training set
  as well as on test sets)
* Application of graph transformations, such as dropout (*limited support*)

In the future we also hope to support:

* Dimension, type and axes-checking

.. image:: https://img.shields.io/coveralls/bartvm/blocks.svg
   :target: https://coveralls.io/r/bartvm/blocks

.. image:: https://travis-ci.org/bartvm/blocks.svg?branch=master
   :target: https://travis-ci.org/bartvm/blocks

.. image:: https://readthedocs.org/projects/blocks/badge/?version=latest
   :target: https://blocks.readthedocs.org/

.. image:: https://img.shields.io/scrutinizer/g/bartvm/blocks.svg
   :target: https://scrutinizer-ci.com/g/bartvm/blocks/

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/bartvm/blocks/blob/master/LICENSE

|

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`

.. UtilsRL documentation master file, created by
   sphinx-quickstart on Tue Jun 14 22:53:28 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to UtilsRL's documentation!
===================================

**UtilsRL** is a Python package designed for agile RL algorithm development. At first it was not intended for providing competitve implementations for baseline RL algorithms, but for fast integration into existing RL code bases and experimenting ideas at least cost. It provides: 

* Basic RL modules
* Experiment management utilities
* Tensorboard loggers
* Tensorboard parsing and visualization utilities

With UtilsRL, you can implement the full version of PPO2 (think about the tricks!) within 200 line of python code. We also tried integrating UtilsRL into other RL frameworks / code bases, and the cost of migration are surprisingly low. 

Currently UtilsRL is maintained by researchers from `Lamda-RL Group <https://github.com/LAMDA-RL>`_, any bug reports / feature requests are welcome and will be dealt with ASAP.

Installation
------------
UtilsRL is currently hosted on `PyPI <https://pypi.org/project/UtilsRL/>`_. It requires Python >= 3.6.
You can simply install UtilsRL from PyPI with:
::

    $ pip install UtilsRL

After Installation, try with:
::

    import UtilsRL
    print(UtilsRL.__version__)

If no error occurs, you have installed UtilsRL successfully.

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials/exp_manage
   tutorials/minimal_ppo
   tutorials/benchmarks


.. toctree::
   :maxdepth: 2
   :caption: Module Design: 

   modules/exp
   modules/rl
   modules/plot
   modules/logger
   modules/monitor


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

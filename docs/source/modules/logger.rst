UtilsRL.logger
==============

This module list the Loggers we provide in `UtilsRL`. \
Among them, :class:`~UtilsRL.logger.TensorboardLogger` is a rather shallow capsulation for ``tensorboard.SummaryWriter``, with enhanced logging strategy and prettier output format.\

The loggers all inherits from ``UtilsRL.logger.BaseLogger`` and are required to implememt ``log_str``. We plan to support for comparatively niche loggers, such as `Aim <https://github.com/aimhubio/aim>`_ in future versions of UtilsRL.

In this page we mainly present the api docs of each logger, and we refer the readers to check their usage in :ref:`Use Loggers`.

UtilsRL.logger.DummyLogger
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: UtilsRL.logger.DummyLogger
    :members:
    :show-inheritance:

UtilsRL.logger.ColoredLogger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: UtilsRL.logger.ColoredLogger
    :members:
    :show-inheritance:

UtilsRL.logger.TensorboardLogger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: UtilsRL.logger.TensorboardLogger
    :members:
    :show-inheritance:
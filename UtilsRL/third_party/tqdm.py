import sys
from time import time
from datetime import datetime, timedelta
from weakref import WeakSet

from tqdm.utils import (
    CallbackIOWrapper, Comparable, DisableOnWriteError, FormatReplace, SimpleTextIOWrapper,
    _basestring, _is_ascii, _range, _screen_shape_wrapper, _supports_unicode, _term_move_up,
    _unich, _unicode, disp_len, disp_trim)
from tqdm import tqdm_notebook
from tqdm import tqdm as tqdm
from tqdm.std import EMA, TqdmTypeError, TqdmKeyError, TqdmWarning, TqdmExperimentalWarning, TqdmMonitorWarning, TqdmDeprecationWarning

__all__ = ["tqdm_tty", "tqdm_notebook", "tqdm_file"]

class tqdm_tty(tqdm):
    def __init__(self, *args, **kwargs):
        super(tqdm_tty, self).__init__(*args, **kwargs)

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = iter(self.iterable)

        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            for obj in iterable:
                yield obj
            return

        if self.initial > 0:
            for _ in range(self.initial):
                next(iterable)

        mininterval = self.mininterval
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        min_start_t = self.start_t + self.delay
        n = self.n
        time = self._time

        try:
            for _ in range(self.total - self.initial):
                yield next(iterable)
                # Update and possibly print the progressbar.
                # Note: does not call self.update(1) for speed optimisation.
                n += 1

                if n - last_print_n >= self.miniters:
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            self.close()

class tqdm_file(tqdm_tty):
    """
    Decorate an iterable_tty object, should be used only when redirecting tty outputs to files.
    """     

    @staticmethod
    def format_meter(n, total, elapsed, ncols=None, prefix='', ascii=False, unit='it',
                     unit_scale=False, rate=None, bar_format=None, postfix=None,
                     unit_divisor=1000, initial=0, colour=None, **extra_kwargs):
        # sanity check: total
        if total and n >= (total + 0.5):  # allow float imprecision (#849)
            total = None

        # apply custom scale if necessary
        if unit_scale and unit_scale not in (True, 1):
            if total:
                total *= unit_scale
            n *= unit_scale
            if rate:
                rate *= unit_scale  # by default rate = self.avg_dn / self.avg_dt
            unit_scale = False

        elapsed_str = tqdm_tty.format_interval(elapsed)

        # if unspecified, attempt to use rate = average speed
        # (we allow manual override since predicting time is an arcane art)
        if rate is None and elapsed:
            rate = (n - initial) / elapsed
        inv_rate = 1 / rate if rate else None
        format_sizeof = tqdm_tty.format_sizeof
        rate_noinv_fmt = ((format_sizeof(rate) if unit_scale else
                           '{0:5.2f}'.format(rate)) if rate else '?') + unit + '/s'
        rate_inv_fmt = (
            (format_sizeof(inv_rate) if unit_scale else '{0:5.2f}'.format(inv_rate))
            if inv_rate else '?') + 's/' + unit
        rate_fmt = rate_inv_fmt if inv_rate and inv_rate > 1 else rate_noinv_fmt

        if unit_scale:
            n_fmt = format_sizeof(n, divisor=unit_divisor)
            total_fmt = format_sizeof(total, divisor=unit_divisor) if total is not None else '?'
        else:
            n_fmt = str(n)
            total_fmt = str(total) if total is not None else '?'

        try:
            postfix = ', ' + postfix if postfix else ''
        except TypeError:
            pass

        remaining = (total - n) / rate if rate and total else 0
        remaining_str = tqdm_tty.format_interval(remaining) if rate else '?'
        try:
            eta_dt = (datetime.now() + timedelta(seconds=remaining)
                      if rate and total else datetime.utcfromtimestamp(0))
        except OverflowError:
            eta_dt = datetime.max

        # format the stats displayed to the left and right sides of the bar
        if prefix:
            # old prefix setup work around
            bool_prefix_colon_already = (prefix[-2:] == ": ")
            l_bar = prefix if bool_prefix_colon_already else prefix + ": "
        else:
            l_bar = ''

        bar = ' {0}Iteration {1}/{2} [{3}<{4},{5}{6}] '.format(
            prefix+": " if prefix else "", n_fmt, total_fmt, elapsed_str, remaining_str, rate_fmt, postfix)
        # bar_len = len(bar)
        # res_len = (ncols - bar_len)//2
        return "{}{}{}".format(
            ">"*10, 
            bar, 
            "<"*10
        )
        
    def __init__(self, *args, **kwargs):
        super(tqdm_file, self).__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        print(tqdm_file.format_meter(**self.format_dict))

    def close(self):
        """Cleanup and (if leave=False) close the progressbar."""
        if self.disable:
            return

        # Prevent multiple closures
        self.disable = True

        # decrement instance pos and remove from internal set
        pos = abs(self.pos)
        self._decr_instances(self)

        if self.last_print_t < self.start_t + self.delay:
            # haven't ever displayed; nothing to clear
            return

        # GUI mode
        if getattr(self, 'sp', None) is None:
            return

        # annoyingly, _supports_unicode isn't good enough
        def fp_write(s):
            self.fp.write(_unicode(s))

        try:
            fp_write('')
        except ValueError as e:
            if 'closed' in str(e):
                return
            raise  # pragma: no cover

        leave = pos == 0 if self.leave is None else self.leave
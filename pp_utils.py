# from __future__ import print_function
# import six

import contextlib
from contextlib import contextmanager
import datetime
import tracemalloc

import numpy as np


def dd(a, showall=False):
    """ Extract only _public_ property names from dir(a). """
    return [x for x in dir(a) if showall or x.find('_') != 0]


def pl(ll):
    """ Print all elements of l, each on a separate line. """
    for x in ll:
        print(x)


def pd(a, *args, **kwargs):
    """
    Print property names of 'a'.

    Additional Args, Kwargs:
        passed to dd (q.v.)
    """
    pl(dd(a, *args, **kwargs))


def str_indent(str_lines,  indent=4):
    """
    Add extra indents into \\n-separated strings.

    Args:

    * str_lines (str):
        A string with multiple lines separated by '\\n's.

    * indent (str or int):
        A prefix string, or number of spaces, to add before each line.

    Returns:
        * result (str):
            A new string with indented lines separated by `\\n's.

    """
    if isinstance(indent, int):
        indent = ' ' * indent
    return '\n'.join([indent + line
                      for line in str_lines.split('\n')])


def inspect_lines(obj, indent=4, showall=False, showcalls=False):
    topline = '<{}>'.format(type(obj))
    attrs = [(name, getattr(obj, name))
             for name in sorted(dir(obj))
             if (showall or not name.startswith('_'))]
    if not showall:
        attrs = [(name, value) for name, value in attrs
                 if not name.startswith('_')]
    if not showcalls:
        attrs = [(name, value) for name, value in attrs
                 if not callable(value)]
    heads = ['{} {:s}'.format(name, str(type(value)))
             for name, value in attrs]
    values = [str(value) if not callable(value) else '()'
              for (name, value) in attrs]
    lines = [str_indent(head + ' : ' + value, indent)
             if len(head + value) < 80
             else str_indent(head + ' :\n'
                             + str_indent(value, indent),
                             indent)
             for head, value in zip(heads, values)]
    return [topline] + lines


def pi(obj, *args, **kwargs):
    """Inspect type+contents of attributes of object.

    Additional and Kwargs:
        passed to inspect_lines (q.v.)
    """
    pl(inspect_lines(obj, *args, **kwargs))


def minmax(v):
    """ Return np.min(v) and np.max(v) as a pair. """
    return (np.min(v), np.max(v))


class TimedBlock(object):
    """A class with contextmanager behaviour, to time a block of statements."""

    def __init__(self, action=None):
        """
        Create contextmanager object.

        After use, this contains the elapsed time result.

        Usage:
          with TimedBlock() as t:
              <statements ...
              ...
              >
          time_taken = t.seconds()

        Note: can use "action" as :
         * True : standard message
         * string : custom message, as in: "print action.format(seconds)"
         * callable : call(self).

        """
        self.start_datetime = None
        self.elapsed_deltatime = None
        if action is None:
            print_call = None
        elif hasattr(action, '__call__'):
            print_call = action
        else:
            if not isinstance(action, str):
                action = 'Time taken : {:12.6f} seconds'

            def _inner_print(self):
                print(action.format(self.seconds()))

            print_call = _inner_print
        self.print_call = print_call

    def __enter__(self):
        self.start_datetime = datetime.datetime.now()
        return self

    def __exit__(self, e1, e2, e3):
        self.end_datetime = datetime.datetime.now()
        self.elapsed_deltatime = self.end_datetime - self.start_datetime
        self._seconds = self.elapsed_deltatime.total_seconds()
        if self.print_call is not None:
            self.print_call(self)

    def seconds(self):
        """Our elapsed time in seconds."""
        return self._seconds


class MemoryMonitor:
    """
    Ultra-simple memory usage tracker.

    Note: these are not nestable -- by starting+stopping the trace, it manages the
    tracked memory usage as a global state.
    """
    def __init__(self):
        self.memory_mb = None

    @contextmanager
    def context(self):
        """Measure the time+memory used by a codeblock."""
        tracemalloc.start()
        yield
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.memory_mb = peak_mem * 1.0 / 1024 ** 2
        self.memory_mib = peak_mem * 1.0e-6


def pwhich(name):
    module = __import__(name)
    print(name, '.__file__ = ', module.__file__)
    print(name, '.__version__ = ', getattr(module, '__version__', '(None)'))


def array_difference_stats(array_a, array_b, min_reldiff=1e-20):
    import numpy.ma as ma
    from collections import OrderedDict
    diffs = array_a - array_b
    absdiffs = np.abs(diffs)
    absmags = 0.5 * (np.abs(array_a) + np.abs(array_b))
    absmags = np.max(absmags, min_reldiff * np.max(absmags))
    reldiffs = absdiffs / absmags
    result = OrderedDict(
        [(name, OrderedDict(
            [(op.__name__, op(array))
             for op in (ma.min, ma.max, ma.mean, ma.median)]))
         for array, name in zip(
            [array_a, diffs, absdiffs, reldiffs],
            ['values', 'diffs', 'abs-diffs', 'rel-diffs'])])
    return result


def format_array_difference_stats(array_a, array_b, **kwargs):
    computed_results = array_difference_stats(array_a, array_b, **kwargs)
    strs = []
    for typename, opsdict in computed_results.items():
        strs.append('{:>10s}:  {}'.format(typename, ',  '.join([
            '{:>6s}={:>12s}'.format(opname, '{:>7g}'.format(value))
            for opname, value in opsdict.items()])))
    return strs


def show_array_difference_stats(array_a, array_b, **kwargs):
    strings = format_array_difference_stats(array_a, array_b, **kwargs)
    print('\n'.join(strings))


def standard_shaped_array(shape):
    # Generate a multidimensional array with standard index-counting values.
    # Examples: 1d = [1, 2, 3], 2d = [[11, 12, 13], [21, 22, 23]],
    # 3d = [[[111], [112]],[121, 122]], [[211], [212]],[221, 222]]]
    ndim = len(shape)
    axis_arrays = []
    for i_dim, dim_len in enumerate(shape):
        axis_array = np.arange(1, dim_len + 1)
        axis_array *= 10 ** (ndim - i_dim - 1)
        axis_array_shape = [1] * ndim
        axis_array_shape[i_dim] = dim_len
        axis_array = axis_array.reshape(axis_array_shape)
        axis_arrays.append(axis_array)
    # Finally, add them all up !
    axis_arrays = np.broadcast_arrays(*axis_arrays)
    result = np.array(axis_arrays).sum(axis=0)
    return result


@contextlib.contextmanager
def debug_print_error_message():
    """
    A context manager to debug-print any error occurring before re-raising it.

    Usage:

        with debug_print_error_message():
            ...

    """
    msg = '\n\n****ERROR MESSAGE:\n{}\n*****\n'
    try:
        yield
    except Exception as e:
        print(msg.format(str(e)))
        raise


@contextlib.contextmanager
def debug_assertRaisesRegexp(self, cls, re, *args, **kwargs):
    """
    A routine which behaves just like 'IrisTest.assertRaisesRegexp',
    but which also debug-prints the error message.

    """
    import iris.tests
    with super(iris.tests.IrisTest, self).assertRaisesRegexp(
            cls, re, *args, **kwargs):
        with debug_print_error_message():
            yield


def show_assertRaisesRegexp_messages(testcase):
    """
    A routine to turn on error message debugging in a test, when called by a
    test in an iris.test.IrisTest testcase class.

    Installs a temporary patch to the assertRaisesRegexp routine, to make it
    debug-print the error message produced.

    Uses IrisTest.patch, so the patch lasts as long as the testcase.
    Can also be called in a 'setUp', to enable it for all testcases.

    Usage:

        def setUp(self):
            ...
            show_assertRaisesRegexp_messages(self)

    """
    testcase.patch('iris.tests.IrisTest.assertRaisesRegexp',
                   debug_assertRaisesRegexp)


class _SlotsHolder(object):
    """
    Abstract parent class for container classes with fixed, named properties.

    Supports: dot-property acccess, comparison, str().

    Inherit + configure by supplying class properties:
    * __slots__ (list of string):
        names of content attributes.
        Order and names also provide object init args and kwargs.
    * _typename : the headline name for the string print
    * _defaults_dict : an kwargs-like dict of defaults for object init.

    """

    def __init__(self, *args, **kwargs):
        """
        Initialise new instance with args or kwargs.

        Args order and Kwargs entries from '__slots__'.

        Unspecified args default to values in 'self._defaults_dict',
        or else None.

        """
        values = getattr(self, '_defaults_dict', {})
        unrecs = [key for key in kwargs if key not in self.__slots__]
        if unrecs:
            unrecs = ', '.join(unrecs)
            msg = 'Unrecognised create kwargs : {}'
            raise ValueError(msg.format(unrecs))
        values.update(kwargs)
        if len(args) > len(self.__slots__):
            msg = 'Number of create args is {} > maximum {}.'
            raise ValueError(msg.format(len(args), len(self.__slots__)))
        values.update(zip(self.__slots__, args))
        for name in self.__slots__:
            setattr(self, name, values.get(name, None))

    def __eq__(self, other):
        matches = [getattr(self, name, None) == getattr(other, name, None)
                   for name in self.__slots__]
        # Include arrays in comparison.
        return all(np.all(match) if hasattr(match, 'dtype') else match
                   for match in matches)

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        msg = "{}({})"
        items = [(name, getattr(self, name)) for name in self.__slots__]
        content = ', '.join('{}={}'.format(name, value)
                            for name, value in items
                            if value is not None)
        typename = getattr(self, '_typename', self.__class__.__name__)
        return msg.format(typename, content)


def test_slotsholder():
    class TstAB(_SlotsHolder):
        __slots__ = ('a', 'b', 'zz')
        _defaults_dict = {'b': 'BB'}
        _typename = 'Slots-AB-tester'

    ab = TstAB()
    assert str(ab) == 'Slots-AB-tester(b=BB)'
    assert ab.a is None
    assert ab.b == 'BB'

    ab.a = 3
    assert str(ab) == 'Slots-AB-tester(a=3, b=BB)'

    ab = TstAB(b=None)
    assert str(ab) == 'Slots-AB-tester()'

    ab = TstAB(7, 4, 22)
    assert str(ab) == 'Slots-AB-tester(a=7, b=4, zz=22)'

    ev = None
    try:
        TstAB(1, 2, 3, 4)
    except ValueError as err:
        ev = err
    assert str(ev) == 'Number of create args is 4 > maximum 3.'

    ev = None
    try:
        TstAB(1, 2, kwoggs=88, b=5, alter=9)
    except ValueError as err:
        ev = err
    assert str(ev) == 'Unrecognised create kwargs : kwoggs, alter'


test_slotsholder()

def ncdump(path_or_pathstr, opts: str ="-h"):
    from os import system as oss
    dumpstr = f'ncdump {opts} {path_or_pathstr!s}'
    oss(dumpstr)

def test_ncdump():
    pth = "/data/users/itpp/git/iris-test-data/test_data/NetCDF/testing/test_monotonic_coordinate.nc"
    ncdump(pth)

# test_ncdump()
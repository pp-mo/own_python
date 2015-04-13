import numpy as np

def dd(a, showall=False):
  """ Extract only _public_ property names from dir(a). """
  return [x for x in dir(a) if showall or x.find('_') != 0]

def pl(l):
  """ Print all elements of l. """
  for x in l:
    print x

def pd(a, *args, **kwargs):
    """
    Print property names of 'a'.

    Additional Args, Kwargs:
        passed to dd (q.v.)
    """
    pl(dd(a, *args, **kwargs))


def str_indent(str_lines, indent=4):
    """Add extra indents into \\n-separated string."""
    indent_str = ' ' * indent
    return '\n'.join([indent_str + line
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
    heads = ['{} {:s}'.format(name, type(value))
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

import datetime
class TimedBlock(object):
    """ A class with contextmanager behaviour, to time a block of statements. """
    
    def __init(self):
        """ Create contextmanager object, which after use contains the elapsed time result. 
        
            Usage:
              with TimedBlock() as t:
                  <statements ...
                  ...
                  >
              time_taken = t.seconds()
        """
        self.start_datetime = None
        self.elapsed_deltatime = None
    
    def __enter__(self):
        self.start_datetime = datetime.datetime.now()
        return self
    
    def __exit__(self, e1, e2, e3):
        self.end_datetime = datetime.datetime.now()
        self.elapsed_deltatime = self.end_datetime - self.start_datetime
        self._seconds = self.elapsed_deltatime.total_seconds()
    
    def seconds(self):
        """ Our elapsed time in seconds. """ 
        return self._seconds

def pwhich(name):
    module = __import__(name)
    print name, '.__file__ = ',module.__file__
    print name, '.__version__ = ', getattr(module, '__version__', '(None)')


def array_difference_stats(array_a, array_b, min_reldiff=1e-20):
    import numpy.ma as ma
    from collections import OrderedDict
    diffs = array_a - array_b
    absdiffs = np.abs(diffs)
    absmags = 0.5*(np.abs(array_a) + np.abs(array_b))
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
    results = array_difference_stats(array_a, array_b, **kwargs)
    strs = []
    for typename, opsdict in results.iteritems():
        strs.append('{:>10s}:  {}'.format(typename, ',  '.join([
                       '{:>6s}={:>12s}'.format(opname, '{:>7g}'.format(value))
                       for opname, value in opsdict.iteritems()])))
    return strs

def show_array_difference_stats(array_a, array_b, **kwargs):
    strings = format_array_difference_stats(array_a, array_b, **kwargs)
    print '\n'.join(strings)


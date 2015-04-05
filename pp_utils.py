import datetime


class TimedBlock(object):
    def __init__(self):
        pass

    def __enter__(self):
        self.start_time = datetime.datetime.now()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = datetime.datetime.now()

    def seconds(self):
        return (self.end_time - self.start_time).total_seconds()


def pl(ll):
    print '\n'.join(str(x) for x in ll)


def dd(x=None, showall=False):
    if x is not None:
        xx = dir(x)
    else:
        xx = dir()
    if not showall:
        xx = [x for x in xx if x[0] != '_']
    return xx


def pd(x=None):
    pl(dd(x))


def exercise_timedblock():
    with TimedBlock() as tb:
        for n in range(1000):
            x = n * n
    print tb.seconds()


if __name__ == '__main__':
    exercise_timedblock()


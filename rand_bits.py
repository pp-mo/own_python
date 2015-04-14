import numpy as np
from numpy.random import uniform

debug = False

def choose_bit(p):
    return uniform() < p

def choose_between(relprobs):
    # make cumulative probs
    levels = np.cumsum(relprobs)
    # normalise so end == 1.0
    levels *= 1.0 / levels[-1]
    # return a random index 0..len(relprobs)-1, weighted choice
    return np.count_nonzero(uniform() > levels)

def test_patch():
    # layout in X/Y
    # [0 1]
    # [2 3]
    mm = np.ones((2, 2, 2, 2,), dtype=float)
    mm[0, 0, 0, 0] = 0
    mm[1, 1, 1, 1] = 0
    mm[1, 0, 0, 1] = 0
    mm[0, 1, 1, 0] = 0
    mm *= 1.0 / np.sum(mm)
    return mm

mm = test_patch()
# [0 1]
# [2 3]

do_tweak_patch = False
do_tweak_patch = True
if do_tweak_patch:
    # adjust patch to favour vertical + horizontal lines
    promote_inds = [
        (1, 1, 0, 0),
        (1, 0, 1, 0),
        (0, 1, 0, 1),
        (0, 0, 1, 1)]
    for inds in promote_inds:
        mm[inds] += 0.3

    # make spaces somewhat likely
    mm[0, 0, 0, 0] = 0.2
    
#     # make 'forbidden' diagonals somewhat possible
#     mm[0, 1, 1, 0] = 0.1
#     mm[1, 0, 0, 1] = 0.1


do_one_zero = False
# do_one_zero = True
if do_one_zero:
    mm[:] = 0.0
    mm[0, 0, 0, 0] = 1.0
    mm[1, 1, 1, 1] = 1.0

mm = mm * 1.0 / np.sum(mm)
print 'matrix:'
print mm * 100.0
print

class OutsideArrayException(Exception):
    pass

def fill_at_patch(bb, ix, iy):
    if ix < 0 or iy < 0 or ix > bb.shape[1]-2 or iy > bb.shape[0]-2:
        raise OutsideArrayException()

    patch = bb[iy:iy+2, ix:ix+2]

    if debug:
        print 'At: y={} x={}'.format(iy, ix)
        def sho(p):
            return ' -' if p is np.ma.masked else '{:02d}'.format(p)
        print 'bb before ='
        print bb
        msg = ('patch before = {:3s} {:3s}\n'
               '               {:3s} {:3s}')
        print msg.format(*[sho(p) for p in patch.flat])

    slices = [slice(None) if pt_val is np.ma.masked else pt_val
              for pt_val in patch.flat]
    probs = list(mm[slices].flat)
    n_bits = sum(pt_val is np.ma.masked for pt_val in patch.flat)
    bits = choose_between(probs)
        # yields an (n_bits)-bit number

    i_bit = n_bits - 1
    for iy1, ix1 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        val = patch[iy1, ix1]
        if val is np.ma.masked:
            assert i_bit >= 0
            bit = (bits & (2 ** i_bit)) != 0
            i_bit -= 1
            if debug:
                print '  [{}] = {}'.format((iy1, ix1), bit)
            patch[iy1, ix1] = bit
            bb[iy+iy1, ix+ix1] = bit

    if debug:
        print 'bb after ='
        print bb
        msg = ('patch after  = {:3s} {:3s}\n'
               '               {:3s} {:3s}')
        print msg.format(*[sho(p) for p in patch.flat])
        print

def fill_array(hx, hy):
    nx = 1 + 2 * (hx + 1)
    ny = 1 + 2 * (hy + 1)
    bb = np.ma.masked_array(np.zeros((ny, nx), dtype=int), mask=True)

#      seed = 7130
#      while 1:
#          np.random.seed(seed)
#          bb = np.ma.masked_array(np.zeros((ny, nx), dtype=int), mask=True)
#          fill_at_patch(bb, hx, hy)
#          if bb[hy, hx+1] == 0 and bb[hy+1, hx+1] == 0:
#              print seed
#              return
#          seed += 1

    steps = spiral_steps(hx, hy)
    limit_steps = 0
    try:
        i_step = 0
        while 1:
            i_step += 1
            if limit_steps and i_step > limit_steps:
                print '(stop)'
                break
            x, y = next(steps)
            fill_at_patch(bb, x, y)
#             print
#             print bb
    except OutsideArrayException:
        pass
    return bb[1:-1, 1:-1]

def spiral_steps(x, y):
    x0 = x1 = x
    y0 = y1 = y
    yield (x, y)
    while 1:
        x1 += 1
        for x in range(x0+1, x1+1):
            yield (x, y)
        y1 += 1
        for y in range(y0+1, y1+1):
            yield (x, y)
        x0 -= 1
        for x in range(x1-1, x0-1, -1):
            yield (x, y)
        y0 -= 1
        for y in range(y1-1, y0-1, -1):
            yield (x, y)

def test_spiral_steps():
    steps = spiral_steps(5, 5)
    a = np.zeros((10, 10), dtype=int)
    for i_step in range(40):
        x, y = next(steps)
        print (x, y)
        a[y, x] = i_step + 100
    print a

def rationalise(mm):
    # average over X and Y reflections
    mm = mm + mm.transpose((1, 0, 3, 2))
    mm = mm + mm.transpose((2, 3, 0, 1))
    # average over 4 rotations
    # 0 1
    # 2 3
    mm = (mm + 
          mm.transpose((2, 0, 3, 1)) +
          mm.transpose((3, 2, 1, 0)) +
          mm.transpose((1, 3, 0, 2)))
    mm = mm * 1.0 / np.sum(mm)
    return mm

def check_rational(mm):
    reorders = [
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (2, 0, 3, 1),
        (3, 2, 1, 0),
        (1, 3, 0, 2)]
    print 'RATIONAL check..'
    for reorder in reorders:
        print 'reorder{} : same={}'.format(
            reorder,
            np.allclose(mm.transpose(reorder), mm))
    return mm

ELEMENTS_INFO = [
    # name, indices, multiplicity
    ('zeros', (0, 0, 0, 0), 1),
    ('ones', (1, 1, 1, 1), 1),
    ('diags', (1, 0, 0, 1), 2),
    ('stripes', (1, 1, 0, 0), 4),
    ('corners', (1, 0, 0, 0), 4),
    ('angles', (1, 1, 1, 0), 4)
]

def describe_elements():
    print 'Debug matrix deconstruction elements:'
    for name, inds, n_ways in ELEMENTS_INFO:
        mmt = np.zeros((2, 2, 2, 2))
        mmt[inds] = 1.0 * n_ways
        mmt = rationalise(mmt)
        print 'elem = {}'.format(name)
        print 100.0 * mmt
        print
    
def deconstruct(mm_in):
    mm = mm_in * 1.0 / np.sum(mm_in)
    def pr(name, keys):
        frac = mm[keys]
        percent = np.round(100.0 * frac, 1)
        print '  {} = {}'.format(name, percent)
    print 'Decode:'
    for name, inds, _ in ELEMENTS_INFO:
        pr(name, inds)

    # Check this info reconstructs the whole thing.
    mm2 = np.zeros((2, 2, 2, 2))
    for name, inds, n_ways in ELEMENTS_INFO:
        mm2[inds] = mm[inds] * n_ways
    mm2 = rationalise(mm2)
#     print 'recons:'
#     print 100.0 * mm2
    assert np.allclose(mm2, mm_in)

def tryseed(seed, cubing=True):
    global mm
    print 'seed = ', seed
    np.random.seed(seed)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 8))
    for i_plt in range(6):
        plt.subplot(231 + i_plt)
#         mm = np.array([ 14.65952644,   5.4971857 ,   5.4971857 ,  10.04062837,
#                        5.4971857 ,  10.04062837,   1.0188372 ,   3.5366293 ,
#                        5.4971857 ,   1.0188372 ,  10.04062837,   3.5366293 ,
#                        10.04062837,   3.5366293 ,   3.5366293 ,  10.21390665]).reshape((2, 2, 2, 2))
        bb = fill_array(30, 20)
#         check_rational(mm)
        print
        print '@', i_plt
        print mm * 100.0
        deconstruct(mm)

        mm = uniform(size=16).reshape((2, 2, 2, 2))
        # "power up" to promote extreme values
        if cubing:
            mm= mm * mm * mm
        else:
            # (square)
            mm= mm * mm
        mm = rationalise(mm)
        plt.pcolormesh(bb, vmin=0, vmax=1)
    plt.show()


if __name__ == '__main__':
    # test_spiral_steps()
    import datetime
    seed = datetime.datetime.now().microsecond

    tryseed(seed, 0)

# some nice ones
#     seed =  570911
#     seed =  744674
#     seed =  811433
#     seed =  162302
# with mm**3 in place of mm**2 ...
#     seed =  933977
#     seed =  296942
#     seed =  640182

    test_seeds_and_cubings = [
        (570911, 0),
        (744674, 0),
        (811433, 0),
        (162302, 0),
        (933977, 1),
        (296942, 1),
        (640182, 1),
    ]
    for seed, cubing in test_seeds_and_cubings:
        tryseed(seed, cubing)

from timeit import timeit
import os.path as op

fn = op.join(op.dirname(__file__), "cat.png")
def timethat(expr, number=1e5, setup='pass', globs=None):
    title = expr
    print('{:40} {:8.5f} µs'.format(title, timeit(expr, number=int(number), globals=globs, setup=setup) * 1000000.0 / float(number)))

num = 5e2
stb_setup = 'import stb.image as im'
pil_setup = 'from PIL import Image'

timethat('x = im.load(fn)', number=num, setup=stb_setup, globs=globals())
timethat('x = Image.open(fn)', number=num, setup=pil_setup, globs=globals())
timethat('x = Image.open(fn).getdata()', number=num, setup=pil_setup, globs=globals())
timethat('x = Image.open(fn).tobytes()', number=num, setup=pil_setup, globs=globals())
timethat('x = np.array(Image.open(fn))', number=num, setup=pil_setup+';import numpy as np', globs=globals())

timethat('y = im.resize(x, 200, 200)', number=num, setup=stb_setup + ';x=im.load(fn)', globs=globals())
timethat('y = x.resize((200, 200))', number=num, setup=pil_setup + ';x=Image.open(fn)', globs=globals())


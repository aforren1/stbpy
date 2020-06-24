from timeit import timeit

def timethat(expr, number=1e5, setup='pass', globs=None):
    title = expr
    print('{:40} {:8.5f} µs'.format(title, timeit(expr, number=int(number), globals=globs, setup=setup) * 1000000.0 / float(number)))

num = 5e2
stb_setup = 'import stb.image as im'
timethat('x = im.load("cat.png")', number=num, setup=stb_setup)

pil_setup = 'from PIL import Image'
timethat('x = Image.open("cat.png")', number=num, setup=pil_setup)
timethat('x = Image.open("cat.png").getdata()', number=num, setup=pil_setup)
timethat('x = Image.open("cat.png").tobytes()', number=num, setup=pil_setup)
timethat('x = np.array(Image.open("cat.png"))', number=num, setup=pil_setup+';import numpy as np')

timethat('y = im.resize(x, 200, 200)', number=num, setup=stb_setup + ';x=im.load("cat.png")')
timethat('y = x.resize((200, 200))', number=num, setup=pil_setup + ';x=Image.open("cat.png")')


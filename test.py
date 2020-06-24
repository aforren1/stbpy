from stb import image as im
#import matplotlib.pyplot as plt

x = im.load('cat.png')
y = im.resize(x, 200, 200)

#plt.imshow(x.reshape(x.shape[1], x.shape[0], x.shape[2]))
#plt.show()

#plt.imshow(y.reshape(y.shape[1], y.shape[0], y.shape[2]))
#plt.show()

z = im.write_png_to_memory(x)
print((x.shape[0] * x.shape[1] * x.shape[2]) - z.shape)

w = im.load_from_memory(z)

print((x == w).all())

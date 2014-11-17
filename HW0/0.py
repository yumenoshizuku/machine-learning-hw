import numpy as np
import matplotlib.pylab as plt
import os
import sys

iris = []
for line in open(os.path.join(sys.path[0], "iris.data"), "r"):
    n = line.split(",")
    if len(n) == 5:
        iris.append(n)
im = np.matrix(iris).transpose()

colors = []
for i in range(50):
    colors.append("r")
for i in range(50):
    colors.append("g")
for i in range(50):
    colors.append("b")

scat = plt.scatter(im[0], im[1], c=colors)
plt.savefig (os.path.join(sys.path[0], "plot01.png"))
scat.remove()
plt.draw()
plt.clf()
scat = plt.scatter(im[0], im[2], c=colors)
plt.savefig (os.path.join(sys.path[0], "plot02.png"))
scat.remove()
plt.draw()
plt.clf()
scat = plt.scatter(im[0], im[3], c=colors)
plt.savefig (os.path.join(sys.path[0], "plot03.png"))
scat.remove()
plt.draw()
plt.clf()
scat = plt.scatter(im[1], im[2], c=colors)
plt.savefig (os.path.join(sys.path[0], "plot12.png"))
scat.remove()
plt.draw()
plt.clf()
scat = plt.scatter(im[1], im[3], c=colors)
plt.savefig (os.path.join(sys.path[0], "plot13.png"))
scat.remove()
plt.draw()
plt.clf()
scat = plt.scatter(im[2], im[3], c=colors)
plt.savefig (os.path.join(sys.path[0], "plot23.png"))
scat.remove()
plt.draw()
plt.clf()
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    return 27 * (1 - x - y) * (x) * (y)


x = np.linspace(0, 1, 4)
y = np.linspace(0, 1, 4)

X, Y = np.meshgrid(x, y)
X = np.tril(X)
Y = np.triu(Y)
print(X, Y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.contour3D(X, Y, Z, 50, cmap="binary")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

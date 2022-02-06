import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera

def f(t,n):
    return (2/n)*(-1)**(n-1)*np.sin(n*t)  #Los coeficientes se encontraron analíticamente

L = np.pi*2
cicles = 2
x = np.linspace(0,L*cicles,1000)

n_init = 1
n_end = 50
F = 0

fig = plt.figure(figsize=(6,6))
camera = Camera(fig)

while n_init <= n_end:
    F += f(x,n_init)
    plt.plot(x,F,c="k")
    camera.snap()
    n_init += 1

animation = camera.animate()
animation.save("Fourier.gif")
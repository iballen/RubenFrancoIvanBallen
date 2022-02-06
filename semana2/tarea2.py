import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
import os.path as path
import wget
import scipy as sp
from scipy import integrate

#Punto 1.2

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

#Punto 1.3

def f_r(t):
    return (1/np.pi)*((t/12)*(t**2-np.pi**2))**2

def GetRiemann_6():
    est, error = integrate.quad(lambda t: f_r(t), -np.pi,np.pi)
    return f"La estimación de zeta(6) es de {est} +/- {error}"

print(GetRiemann_6())

#Punto 1.4

t = np.linspace(-2*np.pi,2*np.pi,100)

def f4(t):
    return np.exp(-0.1*t)*np.sin(t)

def der_ana_f4(t):
    return np.exp(-0.1*t)*(np.cos(t)-0.1*np.sin(t))

def der_right_f4(t):
    return (f4(t)-f4(t-0.1))/(0.1)

X = np.fft.fft(f4(t))
Xmag = np.abs(X)
freq = np.fft.fftfreq(np.size(t),0.02)
X *= 1j*freq
deri_espectral = np.fft.ifft(X)

fig = plt.figure(figsize=(6,6))
plt.plot(t,der_ana_f4(t),color="red",label="derivada analítica")
plt.plot(t,der_right_f4(t),color="blue",label="derivada a derecha")
plt.plot(t,deri_espectral,color="green",label="derivada espectral")
plt.legend()
plt.grid()
plt.savefig("Comparación_derivada.jpg")


#Punto 1.5
Datos_crudos = np.genfromtxt("ManchasSolares.txt")
Datos_semicocinados = list()
for i in range(len(Datos_crudos)):
    if Datos_crudos[i,0] >= 1900:
        Datos_semicocinados.append(Datos_crudos[i])

Datos_semicocinados = np.array(Datos_semicocinados)
Datos_cocidos = list()
contador_manchas = 0
for i in range(len(Datos_semicocinados)):
    año_current = Datos_semicocinados[i,0]
    if i == 0:
        año_past = 1900
    elif año_current == año_past:
        año = año_past + (Datos_semicocinados[i,1]-1)/12
        Datos_cocidos.append([año,Datos_semicocinados[i,3]])
    else:
        año_past = año_current
        Datos_cocidos.append([año_past,Datos_semicocinados[i,3]])


Datos = np.array(Datos_cocidos)
mean_value = np.mean(Datos[:,1])
Datos[:,1] -= mean_value

X = np.fft.fft(Datos[:,1])
freq = np.fft.fftfreq(np.size(X),1/12)

mask = freq < 0
X[mask] = 0

Xmag = np.abs(X/len(X))
ii = np.argmax(np.abs(Xmag))
freq_dom = freq[ii]
mask_2 = freq != freq_dom
X[mask_2] = 0

im_dom = np.fft.ifft(2*X)


fig4 = plt.figure(figsize=(6,6))
plt.title(f"Periodo {round(1/freq_dom,2)} en años")
plt.plot(Datos[:,0],Datos[:,1]+mean_value,label="Manchas solares")
plt.plot(Datos[:,0],im_dom+mean_value,label="Frecuencia dominante", color = "red")
plt.grid()
plt.legend()
plt.savefig("ManchasSolares.jpg")
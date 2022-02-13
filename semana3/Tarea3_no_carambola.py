import numpy as np
import matplotlib.pyplot as plt


#Tiempo libre medio

file = "EnergiaPotencialGas2D.txt"
data = np.loadtxt(file)

valor_medio = np.mean(data[:,1])

data[:,1] -= valor_medio


X = np.fft.fft(data[:,1])
Xmag = np.abs(X)
freq = np.fft.fftfreq(np.size(data[:,1]))

mask_1 = freq < 0
X[mask_1] = 0


ii = np.argmax(Xmag)
fundamental = freq[ii]
tau = 1/fundamental

mask_2 = freq != fundamental
X[mask_2] = 0

inversa = np.fft.ifft(X)



plt.title(f"Tiempo libre medio = {tau} pasos temporales")
plt.plot(data[:,0]*1000,data[:,1],label="Energía Potencial",color="blue")
plt.plot(data[:,0]*1000,np.real(inversa),label="Frecuencia Fundamental",color="red")
plt.legend()
plt.show()


"""
----------------------------------------------------------------------------------------------------------------



----------------------------------------------------------------------------------------------------------------
"""

#Ley de transferencia de calor de Fourier

k = 10
A  = 0.01
l = 0.30
R = 8.31446261815324
c_v = 3/2*R

C = (k*A)/(c_v*l)

def dT_1(T_1,T_2):
    return -C*(T_1-T_2)

def dT_2(T_1,T_2):
    return C*(T_1-T_2)


t = np.linspace(0,100,10000)

def GetRungeKutta4(dT_1,dT_2,T_0,t):
    h = (t[-1]-t[0])/(len(t)-1)

    T_1 = np.zeros(len(t))
    T_2 = np.zeros(len(t))

    T_1[0] = T_0[0]
    T_2[0] = T_0[1]

    K1 = np.zeros(2)
    K2 = np.zeros(2)
    K3 = np.zeros(3)
    K4 = np.zeros(2)

    for i in range(1,len(t)):
        K1[0] = dT_1(T_1[i-1],T_2[i-1])
        K1[1] = dT_2(T_1[i-1],T_2[i-1])

        K2[0] = dT_1(T_1[i-1]+0.5*h, T_2[i-1]+0.5*K1[0]*h)
        K2[1] = dT_2(T_1[i-1]+0.5*h, T_2[i-1]+0.5*K1[1]*h)

        K3[0] = dT_1(T_1[i-1]+0.5*h, T_2[i-1]+0.5*K2[0]*h)
        K3[1] = dT_2(T_1[i-1]+0.5*h, T_2[i-1]+0.5*K2[1]*h)

        K4[0] = dT_1(T_1[i-1]+h,T_2[i-1]+K3[0]*h)
        K4[1] = dT_2(T_1[i-1]+h,T_2[i-1]+K3[1]*h)

        T_1[i] = T_1[i-1] + (h/6)*(K1[0]+2*K2[0]+2*K3[0]+K4[0])
        T_2[i] = T_2[i-1] + (h/6)*(K1[1]+2*K2[1]+2*K3[1]+K4[1])

    return T_1, T_2

T_0 = [400,200]

solucion = GetRungeKutta4(dT_1,dT_2,T_0,t)

plt.figure()
plt.title("Evolución del sistema")
plt.plot(t,solucion[0],label="T_1",color="red")
plt.plot(t,solucion[1],label="T_2",color="blue")
plt.xlabel("Tiempos [s]")
plt.ylabel("Temperatura [K]")
plt.legend()
plt.show()


"""
-------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------
"""
file = "Derivada.txt"
derivada = np.loadtxt(file)

plt.figure()
plt.title("Derivada en C++")
plt.plot(derivada[:,0],derivada[:,1],color = "blue", label="f(x) = -2x*e^(-x^2)")
plt.legend()
plt.grid()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

#PRESENTACIÓN DE FUNCIONES
def f(t,n):
    return 2*(-1)**(n-1)*np.sin(n*t)/n

x=np.linspace(-np.pi,np.pi,1000)
n=1
nt=50
F=0

fig=plt.figure()
camera=Camera(fig)

while n<nt:
    F+=f(x,n)
    plt.plot(x,F,color='k')
    camera.snap()
    n+=1
    
animation=camera.animate()
animation.save('Fourier.gif')

#------------------------------------------------------------------------------



#------------------------------------------------------------------------------

#DERIVADA ESPECTRAL
def h(t):
    return np.exp(-0.1*t)*np.sin(t)

def a_dh(t):
    return -0.1*np.exp(-0.1*t)*np.sin(t)+np.cos(t)*np.exp(-0.1*t)

def d_dh(t):
    return (h(t)-h(t-0.1))/0.1

t=np.linspace(-2*np.pi,2*np.pi,100,endpoint=True)
t_step=(t[1]-t[0])
N=np.size(t)

X=np.fft.fft(h(t))
d_ana=a_dh(t)

XMag = np.abs(X)
freq=np.fft.fftfreq(N,np.max(d_ana)/100)
X*=1j*freq
d_esp=np.fft.ifft(X)

plt.figure()
plt.plot(t,np.real(d_esp),color='b',label='Derivada espectral')
plt.plot(t,d_dh(t),color='k',label='Derivada derecha')
plt.plot(t,a_dh(t),color='r',label='Derivada analítica')
plt.legend()
plt.show()
plt.savefig('Derivada espectral.png')

#------------------------------------------------------------------------------

# MANCHAS SOLARES
file = 'ManchasSolares.dat.txt'
data = np.loadtxt(file)

mask1=data[:,0]>=1900
data_mask1=data[mask1]
x_axis=data_mask1[:,0]+(data_mask1[:,1])/12
data=data_mask1[:,3]-np.mean(data_mask1[:,3])

fft=np.fft.fft(data)
freq=np.fft.fftfreq(len(data),1)
mask = freq < 0
fft[mask] = 0
ffta=np.abs(fft)

ii=np.argmax(ffta)
Ffreq_m=freq[ii]
Ffreq_a=Ffreq_m*12
T=1/Ffreq_a

mask2=freq!=Ffreq_m
fft[mask2]=0
Dom=np.fft.ifft(2*fft)
 
plt.figure()
plt.plot(x_axis,data_mask1[:,3],label='Datos')
plt.plot(x_axis,np.real(Dom)+np.mean(data_mask1[:,3]),label='Frecuencia dominante',color='r')
plt.title(f'Periodo {round(T,2)} en años')
plt.xlabel('Frecuencia [1/mes]')
plt.ylabel('Norma FFT')
plt.legend()
plt.show()
plt.savefig('Manchas Solares.png')

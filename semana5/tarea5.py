import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def GetRK4(q,u0,t):
    h = t[1]-t[0]
    u = np.zeros(len(t))
    u[0] = u0
    for i in range(1,len(t)):
        k1 = u[i-1]**q
        k2 = (u[i-1]+0.5*k1*h)**q
        k3 = (u[i-1]+0.5*k2*h)**q
        k4 = (u[i-1]+k3*h)**q
        u[i] = u[i-1] + h/6*(k1+2*k2+2*k3+k4)
    return u

t = np.linspace(0,10,1000)
Q = [0,0.2,0.4,0.7,1.]

plt.figure()
plt.title("$u' = u^q$")
for q in Q:
    u = GetRK4(q,1,t)
    plt.plot(t,u,label=f"$q={q}$")

plt.xlim(0,10)
plt.ylim(0,30)
plt.legend()
plt.savefig("Ecuacion_no_lineal.png")
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

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
#plt.savefig("Ecuacion_no_lineal.png")

"""
-------------------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------------------------------------
"""
t = sym.Symbol('t',Real=True)
h = sym.Symbol('h',Real=True)

#Coeficientes Adams - Bashforth 3 puntos
F0 = (t-(-h))*(t-(-2*h))/(2*h**2)
F1 = t*(t-(-2*h))/(-h**2)
F2 = t*(t-(-h))/(2*h**2)

I0 = sym.integrate(F0,(t,0,h))
I1 = sym.integrate(F1,(t,0,h))
I2 = sym.integrate(F2,(t,0,h))
print(I0,I1,I2, '\n')

#Coeficientes Adams - Bashforth 4 puntos
F0 = (t-(-h))*(t-(-2*h))*(t-(-3*h))/(6*h**3)
F1 = t*(t-(-2*h))*(t-(-3*h))/(-2*h**3)
F2 = t*(t-(-h))*(t-(-3*h))/(2*h**3)
F3 = t*(t-(-h))*(t-(-2*h))/(-6*h**3)

I0 = sym.integrate(F0,(t,0,h))
I1 = sym.integrate(F1,(t,0,h))
I2 = sym.integrate(F2,(t,0,h))
I3 = sym.integrate(F3,(t,0,h))
print(I0,I1,I2,I3, '\n')

#Coeficientes Adams - Moulton 3 puntos
F0 = t*(t-(-h))/(2*h**2)
F1 = (t-h)*(t-(-h))/(-1*h**2)
F2 = t*(t-h)/(2*h**2)

I0 = sym.integrate(F0,(t,0,h))
I1 = sym.integrate(F1,(t,0,h))
I2 = sym.integrate(F2,(t,0,h))
print(I0,I1,I2, '\n')

#Coeficientes Adams - Moulton 4 puntos
F0 = t*(t-(-h))*(t-(-2*h))/(6*h**3)
F1 = (t-h)*(t-(-h))*(t-(-2*h))/(-2*h**3)
F2 = t*(t-h)*(t-(-2*h))/(2*h**3)
F3 = t*(t-h)*(t-(-h))/(-6*h**3)

I0 = sym.integrate(F0,(t,0,h))
I1 = sym.integrate(F1,(t,0,h))
I2 = sym.integrate(F2,(t,0,h))
I3 = sym.integrate(F3,(t,0,h))
print(I0,I1,I2,I3, '\n')

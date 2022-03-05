from matplotlib import projections
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
plt.savefig("Ecuacion_no_lineal.png")

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

"""
-------------------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Problema gravitacional de los N cuerpos

def norma(vector):
    suma = 0
    for i in range(len(vector)):
        suma += vector[i]**2
    return np.sqrt

class Particle():
    
    def __init__(self, r0,v0,a0,t,m,radius,Id):
        
        self.dt = t[1]-t[0]
        
        self.r = r0
        self.v = v0
        self.a = a0
        
        self.rVector = np.zeros((len(t),len(r0)))
        self.vVector = np.zeros((len(t),len(v0)))
        self.aVector = np.zeros((len(t),len(a0)))
        
        self.m = m
        self.radius = radius
        self.Id = Id
        
        self.MomentumVector = np.zeros((len(t),len(v0)))
        
        self.EpVector = np.zeros((len(t),1))
        self.EkVector = np.zeros((len(t),1))
        
        self.Ep = 0.
        
        self.Force = self.m * self.a


    def CheckForce(self,Lista_particulas):
        self.Force = 0

        for j in range(len(Lista_particulas)):
            P = Lista_particulas[j]
            if self.Id != P.Id:
                self.Force += (-4*np.pi**2)*(self.m*P.m*(self.r-P.r))/(norma(self.r-P.R)**2+0.1**2)**(3/2)

        return None

    def Verlet_Evolution(self,i):
        """
        Antes de ejecutar, asegúrese de que ya ejecutó la función
        CheckForce para garantizar la evolución del método.
        """
        X = self.rVector
        V = self.vVector
        self.r = 2*X[i-1]-X[i-2]+self.a*self.dt**2
        self.v = (self.r-X[i-2])/(2*self.dt)
        self.rVector[i] = self.r
        self.vVector[i] = self.v
        return None


def init_particles(N,m,dt):
    Lista_P = list()
    Nt = int(2/dt)
    t = np.linspace(0,2,Nt)
    for i in range(N):
        es0 = np.random.rand(3) #Vector en esféricas
        es0[0] = 1
        es0[1] *=  np.pi
        es0[2] *= 2*np.pi
        r0 = np.zeros(3)
        r0[0] = es0[0]*np.sin(es0[1])*np.cos(es0[2])
        r0[1] = es0[0]*np.sin(es0[1])*np.sin(es0[2])
        r0[2] = es0[0]*np.cos(es0[1])
        p = Particle(r0,np.zeros(3),np.zeros(3),t,m,0,i)
        Lista_P.append(p)
    return Lista_P

Particulas = np.array(init_particles(100,0.01,0.001))

fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
for i in range(100):
    p = Particulas[i]
    r = p.r
    ax.scatter(r[0],r[1],r[2])

plt.show()

"""
-------------------------------------------------------------------------------------------------------------------------------------------------



-------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Método simpléctico de orden 4

def v_dot(x):
    return -x

def operador(i,h,L,X,v_dot):
    """
    Definimos el operador exponencial simpléctico. Recibe el índice del operador, el paso
    temporal, la letra que codifica que afecta (posición o velocidad), el vector posición 
    velocidad, y la función de aceleración.
    """

    if L == "A":
        x = X[0]
        delta_x = X[1]
        ii = 0

        if i == 1 or i == 4:
            coef = 1/(2*(2-2**(1/3)))

        else:
            coef = (1-2**(1/3))/(2*(2-2**(1/3)))

    if L == "B":
        x = X[1]
        delta_x = v_dot(X[0])
        ii = 1

        if i == 1 or i == 3:
            coef = 1/(2-2**(1/3))

        elif i == 2:
            coef = -(2**(1/3))/(2-2**(1/3))

        else:
            coef = 0

    x += coef*h*delta_x

    X[ii] = x

    return X


t = np.linspace(0,10,1000)
h = t[1]-t[0]
XVector = np.zeros((len(t),2))
XVector[0][0] = 1
XVector[0][1] = 0
codificacion = ["B","A"]
index = [4,3,2,1]

for it in range(len(t)-1):
    X = XVector[it]
    for i in index:
        for L in codificacion:
            X = operador(i,h,L,X,v_dot)
    XVector[it+1] = X

pos = XVector[:,0]
vel = XVector[:,1]

K = 1/2*vel**2
U = 1/2*pos**2

#Ahora, copiamos el código realizado en clase
def GetLeapFrog(r0,t):
    
    N = len(t)
    h = t[1] - t[0]
    
    t1 = np.arange(0 - 0.5*h, 10 - 0.5*h + h, h)
    
    x = np.zeros(N)
    v = np.zeros(N) # El inicializador

    x[0] = r0[0]
    v[0] = r0[1] - 0.5*h*v_dot(x[0])
    
    for i in range(1,N):
        
        v[i] = v[i-1] + h*v_dot(x[i-1])
        x[i] = x[i-1] + h*v[i]
    
    # Igualando los array
    X = np.array([])
    for i in range(len(x)-1):
        X = np.append(X,(x[i]+x[i+1])*0.5)

    V = v[1:]
    
    #print(len(X),len(V))
    
    return X,V

pos_LF, vel_LF = GetLeapFrog(np.array([1.0,0]),t)

K_LF = 0.5*vel_LF**2
U_LF = 0.5*pos_LF**2


plt.title("Conservación en Métodos Simplécticos")
plt.plot(t,K+U,label="Simpléctico Orden 4")
plt.plot(t[0:-1],K_LF+U_LF,label="Simpléctico Orden 2 (LF)")
plt.xlabel("t[s]")
plt.ylabel("E[J]")
plt.ylim(0.4999,0.5001)
plt.legend()
plt.savefig("Métodos simplécticos.png")




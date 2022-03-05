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

"""
Created on Fri Mar  4 15:41:07 2022

@author: rudaf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm
import numba as nb

G=4*np.pi**2
eps=0.1
m=0.01
v0=np.array([0.,0.,0.])
t=np.linspace(0,2,int(2/0.001)+1)
dt=t[1]-t[0]

class Particle():
    def __init__(self,r0,v0,a0,rd,t,m,Id):
        self.r=r0
        self.v=v0
        self.a=a0
        self.rad=rd
        self.m=m
        self.Id=Id
        self.dt=t[1]-t[0]
        self.rVector=np.zeros((len(t),len(r0)))
        self.vVector=np.zeros((len(t),len(v0)))
        self.aVector=np.zeros((len(t),len(a0)))
        
    def SetPosition(self,i,r):
        self.rVector[i]=r
    
    def SetVelocity(self,i,v):
        self.vVector[i]=v
    
    def SetAceleration(self,i,a):
        self.aVector[i]=a
        
    def GetPositionVector(self):
        return self.rVector
    
    def GetRPositionVector(self):
        return self.RrVector
    
    def GetVelocityVector(self):
        return self.vVector
        
    def GetRVelocityVector(self):
        return self.RvVector
    
    def GetRad(self):
        return self.rad
    
    # def Evolution(self,i):        
    #     self.SetPosition(i,self.r)
    #     self.SetVelocity(i,self.v)
        
    #     self.r+=self.v*self.dt
    #     self.v+=self.a*self.dt
                
    def ReduceSize(self,factor):       
        self.RrVector = np.array([self.rVector[0]])
        self.RvVector = np.array([self.vVector[0]])
        
        for i in range(1,len(self.rVector)):
            if i%factor == 0:
                self.RrVector = np.vstack([self.RrVector,self.rVector[i]])
                self.RvVector = np.vstack([self.RvVector,self.vVector[i]])

def norm(v):
    return (v[0]**2+v[1]**2+v[2]**2)**(1/2)
    
    
def ReduceTime(t,factor,parts): 
    pos=[]
    for s in range(len(parts)):
        parts[s].ReduceSize(factor)
       
    Newt = []    
    for i in range(len(t)):
        if i%factor == 0:
            Newt.append(t[i])  
            pos.append(i)
    return np.array(Newt),pos

def GetParticles(N,Limits,Dim,Velo):
    Particles_=[]
    for i in range(N):
        r_0=np.random.uniform(-1.,1.,size=Dim)
        v_0=Velo
        a_0=np.zeros(Dim)
        P=Particle(r_0, v_0, a_0, 0, t, m, i)
        Particles_.append(P)
    return Particles_

Pts=GetParticles(100, 2, 3, v0)
for p in Pts:
    p.SetPosition(0,p.r)
    p.SetPosition(1,p.r+p.v*dt)
    
    
def Simulation():
    for k in tqdm(range(1,len(t)-1)):
        for i in range(len(Pts)):
            Fi=np.array([0.,0.,0.])
            for j in range(len(Pts)):
                if j!=i:
                    Fi-=G*m**2*(Pts[i].GetPositionVector()[k]-Pts[j].GetPositionVector()[k])/(((norm(Pts[i].GetPositionVector()[k]-Pts[j].GetPositionVector()[k]))**2+eps**2)**(3/2))
                    
            ai=Fi/Pts[i].m
            rn=2*Pts[i].GetPositionVector()[k]-Pts[i].GetPositionVector()[k-1]+ai*dt**2
            vn=(rn-Pts[i].GetPositionVector()[k-1])/(2*dt)
            Pts[i].SetPosition(k+1,rn)
            Pts[i].SetAceleration(k,ai)
            Pts[i].SetVelocity(k,vn)
        
Simulation()
            
#Energías 
K=np.zeros(len(t))
U=np.zeros(len(t))
for k in tqdm(range(len(t))):
    kt=0
    ut=0
    for i in range(len(Pts)):
        for j in range(len(Pts)):
            if j!=i:
                ut-=G*m**2/(2*((norm(Pts[i].GetPositionVector()[k]-Pts[j].GetPositionVector()[k]))**2+eps**2)**(3/2))
            kt+=m*norm(Pts[i].GetVelocityVector()[k])**2/2    
    K[k]=kt
    U[k]=ut

E=K+U
# K,U=Simulation()
plt.plot(t,K,'--',color='orange',label='Energía Cinética')
plt.plot(t,U,'g--',label='Energía Potencial')
plt.plot(t,E,'b--',label='Energía Total')
plt.xlabel('t[s]')
plt.ylabel('E[J]')
plt.legend()

redt,pos=ReduceTime(t,15,Pts)
  
        

fig2 = plt.figure(figsize=(5,5))
# ax2 = fig2.add_subplot(1,1,1)
ax2=plt.axes(projection='3d')

def init2():
    ax2.set_xlim(-2,2)
    ax2.set_ylim(-2,2)
    ax2.set_zlim(-2,2)
    

def Update2(i):
    
    plot = ax2.clear()
    init2()
    plot = ax2.set_title(r'$t=%.4f \ seconds$' %(redt[i]), fontsize=15)
    
    for p in Pts:
        x = p.GetRPositionVector()[i,0]
        y = p.GetRPositionVector()[i,1]
        z=p.GetRPositionVector()[i,2]
        if int(p.Id)<len(Pts)/2:
            ax2.scatter3D(x,y,z,color='k')
        else:
            ax2.scatter3D(x,y,z,color='r')
        
    return plot

Animation2 = anim.FuncAnimation(fig2,Update2,frames=len(redt),init_func=init2())

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




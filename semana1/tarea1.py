import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm 
import matplotlib.animation as anim
from mpl_toolkits import mplot3d

class particle():

    def __init__(self,r0,v0,a0,t,m,radius,Id):

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

        self.energia = (1/2)*m*(self.v[0]**2+self.v[1]**2)+m*9.8*self.r[1]
        self.EVector = np.zeros(len(t))

    #Métodos
    def Evolution(self,i):
        self.SetPosition(i,self.r)
        self.SetVelocity(i,self.v)
        self.SetEnergy(i, self.energia)

        #Método de Euler
        self.r += self.v*self.dt
        self.v += self.a*self.dt
        self.energia = 1/2*self.m*(self.v[0]**2+self.v[1]**2)+self.m*9.8*(self.r[1]+20)

    def CheckWallLimits(self,limits,dim,e):

        for i in range(dim):
            if self.r[i] + self.GetRadius() > limits[i]:
                self.v[i] = -e*self.v[i]
            if self.r[i] - self.GetRadius() < -limits[i]:
                self.v[i] = -e*self.v[i]

    def ReduceSize(self,factor):
        self.RrVector = np.array([self.rVector[0]])
        self.RvVector = np.array([self.vVector[0]])
        for i in range(1,len(self.rVector)):
            if i%factor == 0:
                self.RrVector = np.vstack([self.RrVector,self.rVector[i]])
                self.RvVector = np.vstack([self.RvVector,self.vVector[i]])
    #Setters
    def SetPosition(self,i,r):
        self.rVector[i] = r

    def SetVelocity(self,i,v):
        self.vVector[i] = v

    def SetEnergy(self,i,E):
        self.EVector[i] = E

    #Getters

    def GetPosVector(self):
        return self.rVector

    def GetRPosVector(self):
        return self.RrVector

    def GetVelVector(self):
        return self.vVector

    def GetRVelVector(self):
        return self.RvVector

    def GetEVector(self):
        return self.EVector

    def GetRadius(self):
        return self.radius


def RunSimulation_1(t,p,Limits):
    for it in tqdm(range(len(t))):
        p.CheckWallLimits(Limits,len(Limits),0.9)
        p.Evolution(it)
    return p


r0 = np.array([-15.,5.])
v0 = np.array([1.,0.])
a0 = np.array([0.,-9.8])
dt = 0.001
tmax = 40
t = np.arange(0,tmax+dt,dt)
Limits = np.array([20,20])


p = particle(r0,v0,a0,t,1,1,1)

p = RunSimulation_1(t,p,Limits)

#Reducción
def ReduceTime(t,factor):
    p.ReduceSize(factor)
    Newt = list()
    for i in range(len(t)):
        if i%factor == 0:
            Newt.append(t[i])
    return np.array(Newt)

redt = ReduceTime(t,100)


#Animación
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)


def init():
    ax.set_xlim(-Limits[0],Limits[0])
    ax.set_ylim(-Limits[1],Limits[1])

def Update(i):
    plot = ax.clear()
    init()
    plot = ax.set_title(r'$t=%.2f \ seconds$' %(redt[i]), fontsize=15)

    x = p.GetRPosVector()[i,0]
    y = p.GetRPosVector()[i,1]

    vx = p.GetRVelVector()[i,0]
    vy = p.GetRVelVector()[i,1]

    circle = plt.Circle((x,y), p.GetRadius(), color = "k", fill = False)
    plot = ax.add_patch(circle)
    plot = ax.arrow(x,y,vx,vy,color = "r", head_width=0.5)

    return plot

Animation = anim.FuncAnimation(fig,Update,frames=len(redt),init_func=init(), interval=1)
Animation.save('FallingBall.gif', writer="imagemagick")

plt.show()
plt.plot(t,p.GetEVector())
plt.title("Energía Mecánica")
plt.xlabel("Tiempo [s]")
plt.ylabel("Energía [J]")
plt.show()


"""
Con una definición temporal de 0.01, estimamos con la gráfica y la animación obtenida que
alrededor de los 19 segundos la pelota deja de rebotar. Sin embargo, acalaramos que necesitamos
más definición debido a que el resultado obtenido acumuló bastante error proveniente del método
de Euler. En ese orden de ideas, aumentamos la resolución a 0.001 y la reducción en un factor de 100
obteniendo una mejor simulación y estimando que la pelota se detiene entre los 35 y 40 segundos."""

dt = 0.01
tmax = 10
t = np.arange(0,tmax+dt,dt)

#Simulacion 2
def GetParticles(N,Limit,L_Velo,Dim=3,dt=0.01):
    Particles = list()
    for i in range(N):
        r0 = np.random.uniform(-Limit+1.0,Limit-1.0,size=Dim)
        v0 = np.random.uniform(-L_Velo,L_Velo,size=Dim)
        a0 = np.zeros(Dim)
        p = particle(r0,v0,a0,t,1.,1.,i)
        Particles.append(p)
    return Particles

def RunSimulation_2(t,N = 100,L_Velo = 6):
    Particles = GetParticles(N,10,L_Velo)
    for it in tqdm(range(len(t))):
        for i in range(len(Particles)):
            Particles[i].CheckWallLimits([10,10,10],3,1)
            Particles[i].Evolution(it)
    return Particles

Datos_simulacion_2 = RunSimulation_2(t)

def ReduceTimeMultiP(t,factor):
    for p in Datos_simulacion_2:
        p.ReduceSize(factor)
    Newt = list()
    for i in range(len(t)):
        if i%factor == 0:
            Newt.append(t[i])
    return np.array(Newt)

redt_2 = ReduceTimeMultiP(t,10)


#Animación 2
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection="3d")


def init():
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)

def Update(i):
    
    plot = ax.clear()
    init()
    plot = ax.set_title(r'$t=%.2f \ seconds$' %(redt[i]), fontsize=15)
    
    for p in Datos_simulacion_2:
        x = p.GetRPosVector()[i,0]
        y = p.GetRPosVector()[i,1]
        z = p.GetRPosVector()[i,2]
        
        vx = p.GetRVelVector()[i,0]
        vy = p.GetRVelVector()[i,1]
        vz = p.GetRVelVector()[i,2]
        
        ax.scatter3D(x,y,z,color="k")
        
    return plot

Animation = anim.FuncAnimation(fig,Update,frames=len(redt),init_func=init)
Animation.save('Gas.gif', writer="imagemagick")
plt.show()
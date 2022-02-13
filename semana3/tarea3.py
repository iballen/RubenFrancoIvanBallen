import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as anim
from celluloid import Camera


def Magnitude(vector):
    mag = 0
    for i in range(len(vector)):
        mag += vector[i]**2
    return np.sqrt(mag)

def dot(vec_1,vec_2):
    result = 0
    for i in range(len(vec_1)):
        result += vec_1[i]*vec_2[i]
    return result

class Particle():
    """
    En este bloque se define la clase partícula, en el __init__ se definen sus propiedades físicas fundamentales 
    así como el historial de evolución de estas en el tiempo. Para este modelo se agrego la propiedad fuerza
    y su evolución temporal. También se creó la función self.Force(particles) la cual compara la posición de
    self relativa a cada partícula en particles para determinar la fuerza que ejerce cada patícula sobre self.
    """
    
    # init
    def __init__(self, r0,v0,a0,t,m,radius,Id):
        
        self.dt  = t[1] - t[0]
        
        self.r = r0
        self.v = v0
        self.a = a0
        self.F = np.zeros(len(r0))
        self.potential = 0
        
        self.rVector = np.zeros( (len(t),len(r0)) )
        self.vVector = np.zeros( (len(t),len(v0)) )
        self.aVector = np.zeros( (len(t),len(a0)) )
        self.FVector = np.zeros( (len(t),len(r0)) )
        self.potentialVector = np.zeros(len(t))
        self.kineticVector = np.zeros(len(t))
        
        self.m = m
        self.radius = radius
        self.kinetic = 1/2*(self.m)*(Magnitude(self.v))**2
        self.Id = Id
        
    # Method
    def Evolution(self,i):
        
        self.SetPosition(i,self.r)
        self.SetVelocity(i,self.v)
        self.SetAceleration(i,self.a)
        self.SetForce(i,self.F)
        
        
       # print(self.r)
        
        # Euler method
        self.r += self.dt * self.v
        self.v += self.dt * self.a
        self.a = self.F/self.m
        self.kinetic = 1/2*(self.m)*(Magnitude(self.v))**2
    
    def CheckWallLimits(self,limits,dim=2):
        
        for i in range(dim):
            
            if self.r[i] + self.radius > limits[i]:
                self.v[i] = - self.v[i]
            if self.r[i] - self.radius < - limits[i]:
                self.v[i] = - self.v[i]
    
    def Force(self,particles):
        
        #Constante de resistencia a la compresión de las bolitas
        k = 100
        F_vector = np.zeros(len(self.F))
        
        for p in particles:
            dif_pos = self.r - p.r
            dif = Magnitude(dif_pos)
            if dif < self.GetR() + p.GetR():
                F_vector[0] += (k*dif**2)*dif_pos[0]
                F_vector[1] += (k*dif**2)*dif_pos[1]
        self.F = F_vector
            
        
    # Setters
    
    def SetPosition(self,i,r):
        self.rVector[i] = r
        
    def SetVelocity(self,i,v):
        self.vVector[i] = v
        
    def SetAceleration(self,i,a):
        self.aVector[i] = a
        
    def SetForce(self,i,F):
        self.FVector[i] = F

    def SetPotential(self,i,u):
        self.potentialVector[i] = u

    def SetKinetic(self,i,k):
        self.kineticVector[i] = k
        
    # Getters 
    def GetPositionVector(self):
        return self.rVector
    
    def GetRPositionVector(self):
        return self.RrVector 
    

    def GetVelocityVector(self):
        return self.vVector

    def GetRVelocityVector(self):
        return self.RvVector
    
    def GetAcelerationVector(self):
        return self.aVector
    
    def GetR(self):
        return self.radius
    
    def ReduceSize(self,factor):
        
        self.RrVector = np.array([self.rVector[0]]) # initial condition
        self.RvVector = np.array([self.vVector[0]])
        
        
        for i in range(1,len(self.rVector)):
            if i%factor == 0:
                self.RrVector = np.vstack([self.RrVector,self.rVector[i]])
                self.RvVector = np.vstack([self.RvVector,self.vVector[i]])


#Discretización
t_min = 0
t_max = 10
dt = 0.0001

t = np.linspace(t_min,t_max,int((t_max-t_min)/dt))


#Partículas
p_1 = Particle(np.array([0.,-10.]), np.array([20.,0.]), np.zeros(2),t,1,2,1)
p_2 = Particle(np.array([0.,-1.6]), np.zeros(2), np.zeros(2),t,1,2,2)
p_3 = Particle(np.array([-15.,-15.]), np.zeros(2), np.zeros(2),t,1,2,3)

Particles = [p_1,p_2,p_3]


total_kinetic = list()
total_potential = list()

#Simulacion

for it in tqdm(range(len(t))):
    k = 0
    u = 0
    for p in Particles:
        p.Evolution(it)
        p.CheckWallLimits([20,20])
        p.Force(Particles)
        p.potential = -1*dot(p.F,p.rVector[it]-p.rVector[it-1])
        p.SetPotential(it,p.potential)
        p.SetKinetic(it,p.kinetic)
        k += p.kinetic
        u += p.potential
    total_kinetic.append(k)
    total_potential.append(u)


def ReduceTime(t,factor):
    
    for p in Particles:
        p.ReduceSize(factor)
        
    Newt = []
    
    for i in range(len(t)):
        if i%factor == 0:
            Newt.append(t[i])
            
    return np.array(Newt)

redt = ReduceTime(t,100)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.set_xlim(-20,20)
ax.set_ylim(-20,20)
camera = Camera(fig)

for i in tqdm(range(len(redt))):
    for p in Particles:
        x = p.GetRPositionVector()[i,0]
        y = p.GetRPositionVector()[i,1]
        
        vx = p.GetRVelocityVector()[i,0]
        vy = p.GetRVelocityVector()[i,1]
        
        circle = plt.Circle( (x,y), p.GetR(), color='k', fill=False)
        plot = ax.add_patch(circle)
        plot = ax.arrow(x,y,vx,vy,color='r',head_width=0.5)
        
    camera.snap()


total_kinetic = np.array(total_kinetic)
total_potential = np.array(total_potential)

plt.figure()
plt.plot(t,total_kinetic,label="cinética")
plt.plot(t,total_potential,label="potencial")
plt.plot(t,total_kinetic+total_potential,label="total")
plt.legend()
plt.show()



animation = camera.animate()
animation.save('Carambola.gif')

"""
------------------------------------------------------------------------------------------------------------------------------------




------------------------------------------------------------------------------------------------------------------------------------
"""

name = "EnergiaPotencialGas2D.txt"
data = np.loadtxt(name)
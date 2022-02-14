# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 00:16:22 2022

@author: rudaf
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from celluloid import Camera

class Particle():
    def __init__(self,r0,v0,a0,rd,t,m,Id):
        self.r=r0
        self.v=v0
        self.a=a0
        self.rad=rd
        self.m=m
        self.Id=Id
        self.dt=t[1]-t[0]
        self.k=m*np.sum(v0**2)/2
        self.rVector=np.zeros((len(t),len(r0)))
        self.vVector=np.zeros((len(t),len(v0)))
        self.aVector=np.zeros((len(t),len(a0)))
        self.kVector=np.zeros(len(t))
        
    def SetPosition(self,i,r):
        self.rVector[i]=r
    
    def SetVelocity(self,i,v):
        self.vVector[i]=v
    
    def SetAceleration(self,i,a):
        self.aVector[i] = a
        
    def SetKinetic(self,i,k):
        self.kVector[i]=k
        
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
    
    def GetRAcelerationVector(self):
        return self.RaVector
    
    def GetRad(self):
        return self.rad
    
    def Evolution(self,i):        
        self.SetPosition(i,self.r)
        self.SetVelocity(i,self.v)
        self.SetAceleration(i,self.a)
        
        self.k=self.m*np.sum(self.v**2)/2
        self.SetKinetic(i, self.k)
        
        self.v+=self.a*self.dt
        self.r+=self.v*self.dt
        
        self.F=0.
        self.a=np.zeros(2)
                        
    def CheckWallLimits(self,limits,dim,e):        
        for i in range(dim):
            if self.r[i] + self.rad > limits[i] and self.v[i]>=0:
                self.v[i] = - self.v[i]*e
            elif self.r[i] - self.rad < - limits[i] and self.v[i]<=0:
                self.v[i] = - self.v[i]*e
                
    def ReduceSize(self,factor):       
        self.RrVector = np.array([self.rVector[0]])
        self.RvVector = np.array([self.vVector[0]])
        self.RkVector=np.array([self.kVector[0]])
        
        for i in range(1,len(self.rVector)):
            if i%factor == 0:
                self.RrVector = np.vstack([self.RrVector,self.rVector[i]])
                self.RvVector = np.vstack([self.RvVector,self.vVector[i]])
                self.RkVector = np.vstack([self.RkVector,self.kVector[i]])
                
    def a_change_hitting(self,Particles):
        F_vector = np.zeros(len(self.a))
        for p in Particles:
            if self.Id!=p.Id:
                dif_pos = np.sum((self.r - p.r)[:]**2)**(1/2)
                if dif_pos < self.GetRad() + p.GetRad():
                    F_vector[0] += (100*dif_pos**2)*(self.r - p.r)[0]
                    F_vector[1] += (100*dif_pos**2)*(self.r - p.r)[1]
        
        self.a=F_vector/self.m
                
        
def ReduceTime(t,factor,parts): 
    for s in range(len(parts)):
        parts[s].ReduceSize(factor)
       
    Newt = []    
    for i in range(len(t)):
        if i%factor == 0:
            Newt.append(t[i])            
    return np.array(Newt)

def Interactions(Ps,id1):
    Ps[id1].a_change_hitting(Ps)   
    return Ps

            
tf=10
dt=1e-4
t=np.arange(0,tf+dt,dt)
P1=Particle(np.array([-10.,1.]), np.array([20.,0.]), np.array([0.,0.]), 2, t, 1, 0)
P2=Particle(np.array([0.,-1.6]), np.array([0.,0.]), np.array([0.,0.]), 2, t, 1, 1)
P3=Particle(np.array([-15.,-15.]), np.array([0.,0.]), np.array([0.,0.]), 2, t, 1, 2)
Ps=np.array([P1,P2,P3])
limits1=[20,20]
N=len(Ps)

def RunSimulation(t,Ps):
    for i in tqdm(range(len(t))):
        for j in range(N):
            Ps=Interactions(Ps,j)
            Ps[j].CheckWallLimits(limits1,2,1)           
            Ps[j].Evolution(i)
            
    return Ps

Pts=RunSimulation(t,Ps)
redt=ReduceTime(t,100,Pts)

K=Pts[0].kVector+Pts[1].kVector+Pts[2].kVector
mask=K>200
K[mask]=200
P=np.abs(K-K[0])
T=K+P

plt.figure()
plt.plot(t,K,'--',label='Energía cinética')
plt.plot(t,P,'--',label='Energía potencial')
plt.plot(t,T,'-',label='Energía total')
plt.xlabel('t[s]')
plt.ylabel('E[J]')
plt.legend()
plt.show()
plt.savefig('Energías.png')
            
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.set_xlim(-20,20)
ax.set_ylim(-20,20)
camera = Camera(fig)

colors=['y','b','k']
labels=['Pelotita 1','Pelotita 2','Pelotita 3']
for i in tqdm(range(len(redt))):   
    for p in range(len(Pts)):
        x = Pts[p].GetRPositionVector()[i,0]
        y = Pts[p].GetRPositionVector()[i,1]
        
        vx = Pts[p].GetRVelocityVector()[i,0]
        vy = Pts[p].GetRVelocityVector()[i,1]
        
        circle = plt.Circle( (x,y), Pts[p].GetRad(), color=colors[p], fill=True)
        plot = ax.add_patch(circle)
        plot = ax.arrow(x,y,vx,vy,color='r',head_width=0.5)
        
    camera.snap()
    
animation = camera.animate(interval=2)
animation.save('Carambola.gif')


        
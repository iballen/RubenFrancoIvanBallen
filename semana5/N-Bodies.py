# -*- coding: utf-8 -*-
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
        
        
            
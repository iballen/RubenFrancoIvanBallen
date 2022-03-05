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
    K=np.zeros(len(t))
    U=np.zeros(len(t))
    P=[]
    L=[]
    for k in tqdm(range(1,len(t)-1)):       
        kt=0
        ut=0
        p=np.zeros(3)
        l=np.zeros(3)
        for i in range(len(Pts)):
            Fi=np.array([0.,0.,0.])
            for j in range(len(Pts)):
                if j!=i:
                    Fi-=G*m**2*(Pts[i].GetPositionVector()[k]-Pts[j].GetPositionVector()[k])/(((norm(Pts[i].GetPositionVector()[k]-Pts[j].GetPositionVector()[k]))**2+eps**2)**(3/2))
                    ut-=G*m**2/(2*((norm(Pts[i].GetPositionVector()[k-1]-Pts[j].GetPositionVector()[k-1]))**2+eps**2)**(3/2))
                kt+=m*norm(Pts[i].GetVelocityVector()[k-1])**2/2 
            p+=m*Pts[i].GetVelocityVector()[k-1]
            l+=np.cross(m*Pts[i].GetPositionVector()[k],Pts[i].GetVelocityVector()[k-1])
            ai=Fi/Pts[i].m
            rn=2*Pts[i].GetPositionVector()[k]-Pts[i].GetPositionVector()[k-1]+ai*dt**2
            vn=(rn-Pts[i].GetPositionVector()[k-1])/(2*dt)
            Pts[i].SetPosition(k+1,rn)
            Pts[i].SetAceleration(k,ai)
            Pts[i].SetVelocity(k,vn)
        l=norm(l)
        K[k]=kt
        U[k]=ut
        P.append(p)
        L.append(l)
    return K,U,P,L       
K,U,P,L=Simulation()

E=np.array([U[0]]*(len(U)))
K=-U
E=U+K

px=np.zeros(len(P))
py=np.zeros_like(px)
pz=np.zeros_like(px)
for i in range(len(P)):
    px[i]=P[i][0]
    py[i]=P[i][1]
    pz[i]=P[i][2]

def graph(x,y,tl,c,lab,xlab,ylab):
    plt.plot(x,y,tl,color=c,label=lab)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()

#Energies
plt.figure(figsize=(5,5))
graph(t,K,'--','orange','Energía Cinética','t[years]','E[J]')
graph(t,U,'--','green','Energía Potencial','t[years]','E[J]')
graph(t,E,'-','#1E90FF','Energía Total','t[years]','E[J]')
plt.savefig('Energies.png')

#Linear momentum
plt.figure(figsize=(5,5))
graph(t[:1999],px,'-','#1E90FF','Momento en x','t[years]','P[kg*v]')
graph(t[:1999],py,'-','orange','Momento en y','t[years]','P[kg*v]')
graph(t[:1999],pz,'-','green','Momento en z','t[years]','P[kg*v]')
plt.ylim(-1,1)
plt.savefig('Linear momentum.png')

#Angular momentum
plt.figure(figsize=(5,5))
graph(t[:1999],L,'-','#1E90FF','Magnitude of the total angular momentum','t[years]','|L|[]')
plt.ylim(-0.02,0.02)
plt.savefig('Angular momentum.png')
    

redt,pos=ReduceTime(t,10,Pts)      

fig = plt.figure(figsize=(5,5))
ax2=plt.axes(projection='3d')

def init():
    ax2.set_xlim(-2,2)
    ax2.set_ylim(-2,2)
    ax2.set_zlim(-2,2)
    

def Update(i):
    
    plot = ax2.clear()
    init()
    plot = ax2.set_title(r'$t=%.4f \ years$' %(redt[i]), fontsize=15)
    
    for p in Pts:
        x = p.GetRPositionVector()[i,0]
        y = p.GetRPositionVector()[i,1]
        z=p.GetRPositionVector()[i,2]
        if int(p.Id)<len(Pts)/2:
            ax2.scatter3D(x,y,z,color='k')
        else:
            ax2.scatter3D(x,y,z,color='r')
        
    return plot

Animation = anim.FuncAnimation(fig,Update,frames=len(redt),init_func=init(),interval=1)
Animation.save('N-bodies.gif', writer='imagemagick')
        
        
            
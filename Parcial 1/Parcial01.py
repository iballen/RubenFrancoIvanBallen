# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:06:24 2022

@author: rudaf
"""
import numpy as np
import matplotlib.animation as anim
import matplotlib.pyplot as plt

k = 5 
l = 3 
m = 2 
g = 9.8 
r0 = 15 
theta0 = np.pi/8
v0 = 0
w0 = 0

def ar(k,m,l,r,g,theta,w):
    return k*(l-r)/m+g*np.cos(theta)+r*w**2

def aa(g,r,theta,v,w):
    return -1*g*np.sin(theta)/r-2*v*w/r

def Get_euler(x0,x0_dot,h):
    return x0+h*x0_dot

def evolution_pos(r,r_dot,r_2dot,r_2dotP,h):
    return r+h*r_dot+(1/6)*(4*r_2dot-r_2dot)*h**2

def evolution_vel(r_dot,r_2dot,r_2dotP,h):
    return r_dot + 0.5*(3*r_2dot-r_2dotP)*h

ar0=ar(k,m,l,r0,g,theta0,w0)
aa0=aa(g,r0,theta0,v0,w0)

t=np.linspace(0,30,2001)
h=t[1]-t[0]

R=np.zeros(len(t))
THETA=np.zeros(len(t))
V=np.zeros(len(t))
W=np.zeros(len(t))
AR=np.zeros(len(t))
AA=np.zeros(len(t))

R[0]=r0
THETA[0]=theta0
AR[0]=ar0
AA[0]=aa0

V[1]=Get_euler(V[0],ar0,h)
W[1]=Get_euler(W[0],aa0,h)
R[1]=Get_euler(R[0],V[1],h)
THETA[1]=Get_euler(THETA[0],W[1],h)


for i in range(1,len(t)-1):
    R[i+1]=evolution_pos(R[i],V[i],AR[i],AR[i-1],h)
    THETA[i+1]=evolution_pos(THETA[i],W[i],AA[i],AA[i-1],h)
    V[i+1]=evolution_vel(V[i],AR[i],AR[i-1],h)
    W[i+1]=evolution_vel(W[i],AA[i],AA[i-1],h)
    AR[i+1]=ar(k,m,l,R[i+1],g,THETA[i+1],W[i+1])
    AA[i+1]=aa(g,R[i+1],THETA[i+1],V[i+1],W[i+1])

Rc=np.zeros(len(t))
THETAc=np.zeros(len(t))
Vc=np.zeros(len(t))
Wc=np.zeros(len(t))

Rc[0]=r0
THETAc[0]=theta0
Vc[1]=Get_euler(V[0],ar0,h)
Wc[1]=Get_euler(W[0],aa0,h)
Rc[1]=Get_euler(R[0],V[1],h)
THETAc[1]=Get_euler(THETA[0],W[1],h)

for i in range(1,len(t)-1):
    Vc[i+1]=Vc[1]+(5*AR[i+1]+8*AR[i]-AR[i-1])*h/12
    Wc[i+1]=Wc[1]+(5*AA[i+1]+8*AA[i]-AA[i-1])*h/12
    Rc[i+1]=evolution_pos(Rc[i],Vc[i],AR[i],AR[i-1],h)
    THETAc[i+1]=evolution_pos(THETAc[i],Wc[i],AA[i],AA[i-1],h)
    
pos=[]
for i in range(len(t)):
    if i%20==0:
        pos.append(i)

rR=R[pos]
rTHETA=THETA[pos]
rV=V[pos]
rW=W[pos]
rAR=AR[pos]
rAA=AA[pos]

rRc=Rc[pos]
rTHETAc=THETAc[pos]
rVc=Vc[pos]
rWc=Wc[pos]

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1,1,1)

def init1():
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    ax.grid()
    
def Update1(i): 
    plot = ax.clear()
    init1()
    x=rR[i]*np.cos(rTHETA[i]-np.pi/2)
    y=rR[i]*np.sin(rTHETA[i]-np.pi/2)
    circle = plt.Circle( (x,y), 2, color='k', fill=True) 
    spring = plt.arrow(0, 0, x, y, color='r')
    plot = ax.add_patch(circle)
    plot = ax.add_patch(spring)
    return plot
Animation = anim.FuncAnimation(fig,Update1,frames=len(pos),init_func=init1)
Animation.save('Péndulo parcial.gif', writer='imagemagick')

fig1=plt.figure(figsize=(5,5))
ax1=fig1.add_subplot(1,1,1,projection='polar')
ax1.plot(rTHETA,rR,'.')
plt.savefig('PolarProyection')

# def init2():   
#     ax1.set_rmax(np.max(rR))
    
# def Update2(i):
#     plot=ax1.clear()
#     init2()
#     r=rR[i]
#     t=rTHETA[i]
#     polar=ax1.scatter(t,r)
#     plot =ax1.add_patch(polar)
#     return plot
    
# Animation2 = anim.FuncAnimation(fig1,Update2,frames=len(pos),init_func=init2())
   

    
    





import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm 
import matplotlib.animation as anim

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

    #Métodos
    def Evolution(self,i):
        self.SetPosition(i,self.r)
        self.SetVelocity(i,self.v)

        #Método de Euler
        self.r += self.v*self.dt
        self.v += self.a*self.dt

    #Setters
    def SetPosition(self,i,r):
        self.rVector[i] = r

    def SetVelocity(self,i,v):
        self.vVector[i] = v

    #Getters

    def GetPosVecotr(self):
        return self.rVector

    def GetVelVector(self):
        return self.vVector
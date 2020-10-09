# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 20:40:31 2020

@author: ASUS
"""
import numpy as np

#Exercise 4
y=np.array([[13,14,16]])
x=np.matrix('1 24; 1 30; 1 36')

#calculate B
y_T=y.transpose()
x_T=x.transpose()
step1=np.linalg.inv(np.matmul(x_T,x))
step2=np.matmul(x_T,y_T)
B=np.matmul(step1,step2)

#Calculate SSE
SSE=np.linalg.norm(y_T-np.matmul(x,B), ord=2)**2
print(SSE)

#Exercise 5
x=np.matrix('1 25; 1 34')
y_hat=np.matmul(x,B)
print(y_hat)

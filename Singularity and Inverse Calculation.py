import numpy as np
import math
import sympy as sp


# link lengths in mm
a1 = float(input("a1 = "))
a2 = float(input("a2 = "))
a3 = float(input("a3 = "))

# joint variables: is mm if f, is degrees if theta\

T1 = float(input("T1 = ")) 
T2 = float(input("T2 = ")) 
d3 = float(input("d3 = ")) 

# degree to radian
T1 = (T1/180.0)*np.pi
T2 = (T2/180.0)*np.pi

# Parametric Table (theta, alpha, r, d)
PT = [[T1,(90.0/180.0)*np.pi,0,a1],
      [T2 + (90.0/180.0)*np.pi,(90.0/180.0)*np.pi,0,0],
      [0,0,0,a2 + a3 + d3]]


# HTM formulae
i = 0
H0_1 = [[np.cos(PT[i][0]),-np.sin(PT[i][0])*np.cos(PT[i][1]),np.sin(PT[i][0])*np.sin(PT[i][1]),PT[i][2]*np.cos(PT[i][0])],
        [np.sin(PT[i][0]),np.cos(PT[i][0])*np.cos(PT[i][1]),-np.cos(PT[i][0])*np.sin(PT[i][1]),PT[i][2]*np.sin(PT[i][0])],
        [0,np.sin(PT[i][1]),np.cos(PT[i][1]),PT[i][3]],
        [0,0,0,1]]

i = 1
H1_2 = [[np.cos(PT[i][0]),-np.sin(PT[i][0])*np.cos(PT[i][1]),np.sin(PT[i][0])*np.sin(PT[i][1]),PT[i][2]*np.cos(PT[i][0])],
        [np.sin(PT[i][0]),np.cos(PT[i][0])*np.cos(PT[i][1]),-np.cos(PT[i][0])*np.sin(PT[i][1]),PT[i][2]*np.sin(PT[i][0])],
        [0,np.sin(PT[i][1]),np.cos(PT[i][1]),PT[i][3]],
        [0,0,0,1]]

i = 2
H2_3 = [[np.cos(PT[i][0]),-np.sin(PT[i][0])*np.cos(PT[i][1]),np.sin(PT[i][0])*np.sin(PT[i][1]),PT[i][2]*np.cos(PT[i][0])],
        [np.sin(PT[i][0]),np.cos(PT[i][0])*np.cos(PT[i][1]),-np.cos(PT[i][0])*np.sin(PT[i][1]),PT[i][2]*np.sin(PT[i][0])],
        [0,np.sin(PT[i][1]),np.cos(PT[i][1]),PT[i][3]],
        [0,0,0,1]]

H0_1 = np.matrix(H0_1)
H1_2 = np.matrix(H1_2)
H2_3 = np.matrix(H2_3)

H0_2 = np.dot(H0_1,H1_2)
H0_2 = np.array(H0_2)
H0_3 = np.dot(H0_2,H2_3)
H0_3 = np.array(H0_3)
#print("H0_3= ")


## ------------------------------------------------------------##

## Jacobian Matrix

Z_1 = [[0],[0],[1]]  # [0,0,1] vector

## row 1 - 3 column 1

J1a = [[1,0,0],
      [0,1,0],
      [0,0,1]]
J1a = np.dot(J1a,Z_1)

J1b_1 = H0_3[0:3 , 3:]
J1b_1 = np.array(J1b_1)

J1b_2 = [[0],
         [0],
         [0]]
J1b_2 = np.array(J1b_2)

J1b = J1b_1 - J1b_2

J1 = [[J1a[1,0]*J1b[2,0]-J1a[2,0]*J1b[1,0]],
      [J1a[2,0]*J1b[0,0]-J1a[0,0]*J1b[2,0]],
      [J1a[0,0]*J1b[1,0]-J1a[1,0]*J1b[0,0]]]

J1 = np.array(J1)

## row 1 - 3 column 2

J2a = H0_1[0:3,0:3]
J2a = np.dot(J2a,Z_1)

J2b_1 = H0_3[0:3 , 3:]
J2b_1 = np.array(J2b_1)

J2b_2 = H0_1[0:3 , 3:]
J2b_2 = np.array(J2b_2)

J2b = J2b_1 - J2b_2

J2 = [[J2a[1,0]*J2b[2,0]-J2a[2,0]*J2b[1,0]],
      [J2a[2,0]*J2b[0,0]-J2a[0,0]*J2b[2,0]],
      [J2a[0,0]*J2b[1,0]-J2a[1,0]*J2b[0,0]]]
J2 = np.array(J2)

## row 1 - 3 column 3

J3 = H0_2[0:3,0:3]
J3 = np.dot(J3,Z_1)


print(J1)
print(J2)
print(J3)



## row 4-6 column 1

J4 = J1a
J4 = np.array(J4)

## row 4-6 column 2

J5 = J2a
J5 = np.array(J5)

## row 4-6 column 3


J6 = [[0],[0],[0]]
J6 = np.array(J6)


print(J4)
print(J5)
print(J6)



JM1 = np.concatenate((J1,J2,J3),1)
print("Jacobian Linear = ")
print(np.around(JM1, 3))

JM2 = np.concatenate((J4,J5,J6),1)
print("Jacobian Rotational = ")
print(np.around(JM2, 3))

J = np.concatenate((JM1,JM2),0)
J = np.around(J,3)


print(J)


## differential equations

xp, yp, zp = sp.symbols('x* y* z*')
ωx, ωy, ωz = sp.symbols('ωx ωy ωz')
T1_p, T2_p, d3_p = sp.symbols('T1* T2* d3*')

q = [[T1_p],[T2_p],[d3_p]]


E = np.dot(J,q)
E = np.array(E)

print(E)

xp = E[0,0]
yp = E[1,0]
zp = E[2,0]
ωx = E[3,0]
ωy = E[4,0]
ωz = E[5,0]

print("xp = ", xp)
print("yp = ", yp)
print("zp = ", zp)
print("ωx = ", ωx)
print("ωy = ", ωy)
print("ωz = ", ωz)


## singularity

D_J = np.linalg.det(JM1)
print(D_J)

## Inverse Velocity

I_V = np.linalg.inv(JM1)
print =(I_V)

## force torque analysis

F_T = np.transpose(JM1)

print(JM1)
print(F_T)

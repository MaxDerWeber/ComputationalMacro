import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# parameter specifications
sigma= 2
kappa=0.3
beta=0.99
phi_1=1.5
phi_2=0.2


A = np.array([[1, 0, 1/sigma], 
                [-kappa, 1, 0],
                [0, 0, 1]])
A_inv = np.linalg.inv(A)

#1

def shock_model1(expec_vec):
    B_shocked = np.zeros((3,2))
    B_ex = np.zeros((3,2)) # expected values

    expY = expec_vec[0]
    expPi = expec_vec[1]
    expi = expec_vec[2]


    e1 = expec_vec[3]
    e2 = expec_vec[4]
    e3 = expec_vec[5] #in c) it is 1

    # B as 3x2 matrix instead of 3x1
    # expected values, shocks with mean 0
    B_ex[0] = np.array([expY + 1/sigma * expPi, 0])
    B_ex[1] = np.array([beta * expPi, 0])
    B_ex[2] = np.array([phi_1 * expPi + phi_2 * expY, 0])
    # with realized shock
    B_shocked[0] = np.array([expY + 1/sigma * expPi, e1])
    B_shocked[1] = np.array([beta * expPi, e2])
    B_shocked[2] = np.array([phi_1 * expPi + phi_2 * expY, e3])

    C_ex = np.dot(A_inv,B_ex)
    C_shocked = np.dot(A_inv,B_shocked)
    diff = C_ex - C_shocked
    diff = diff.reshape(6,)
    return diff

#2
init = [1,0,1,1,0,0]
[coefficients, inf, ier, msg] = fsolve(shock_model1, init, full_output = True)
print(coefficients,msg)

#3 
init3 = [1,0,1,1,0,1]
[coefficients3, inf, ier, msg3] = fsolve(shock_model1, init, full_output = True)
print(coefficients3,msg3)
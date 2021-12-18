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
    phi_1=1.5
    phi_2=0.2

    B_shocked = np.zeros((3,2))
    B_ex = np.zeros((3,2)) # expected values

    expY = expec_vec[0]
    expPi = expec_vec[1]
    expi = expec_vec[2]

    e1 = expec_vec[3]
    e2 = expec_vec[4]
    e3 = expec_vec[5] #in 3) it is 1

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


# takeaways from the solution:
    # The input needs to be effect from static variable and effect from the shocks, not the shocks themselves
    # Also it is said, that in 3. the input vector has length 9, which is not the case in my solution
    # the output then needs to be the three effect-size variables from the model with [0:3] and without [3:6] the shocks




#### Solution from the lecture:

# 1.
def statefunc(coef):
    phi1=1.5
    phi2=0.2

    C_outcome = np.zeros(len(coef))
    c0 = C_outcome[0:3]
    c1 = C_outcome[3:6] # not used in 1)

    # The constants and the shocks both show in the msv
    B = np.zeros(3)

    # First we calculate the contribution of the constants to the model outcome
    expY = c0[0]
    exPi = c0[1]
    shockX = 0


    B[0] = expY + 1/sigma * exPi + shockX
    B[1] = beta * exPi
    B[2] = phi1 * exPi + phi2 * expY

    C_outcome[0:3] = np.dot(A_inv,B)

    # Next, we calculate the contribution of the shock to the model:
    B = np.zeros(3)
    expY = c0[0]
    exPi = c0[1]

    # the interpretation of C1 is the effect of a shock of size 1 on z_t
    shockX = 1

    B[0] = expY + 1/sigma * exPi + shockX
    B[1] = beta * exPi
    B[2] = phi1 * exPi + phi2 * expY

    C_outcome[3:6] = np.dot(A_inv, B) - C_outcome[0:3]
    equations = C_outcome - np.array(coef)
    return equations

# 2.
init = [0] * 6
[msv_coefficients, inf, ier, msg] = fsolve(statefunc, init, full_output = True)
if ier != 0:
    print(msg)

msv_constants = msv_coefficients[0:3]
msv_shock_Y_coefs = msv_coefficients[3:6]

print('The constants in the msv solution are: ', msv_constants)
print('The coefficients on the output gap shock are: ', msv_shock_Y_coefs)

# 3.
pos = 0

def statefunc(coef):
    C_outcome = np.zeros(len(coef))
    c0 = C_outcome[0:3]
    c1 = C_outcome[3:6]
    c2 = C_outcome[6:9]

    phi1=1.5
    phi2=0.2

    for [shock_Y, shock_i] in [[0,0], [1,0], [0,1]]:
        expY = c0[0]
        exPi = c0[1]

        B = np.zeros(3) # initialize for each iteration

        # r_bar, X_target, Pi_target are constant and should aonly be considered when looking at constants (for later???)
        B[0] = expY + 1/sigma * exPi + shock_Y
        B[1] = beta * exPi
        B[2] = phi1 * exPi + phi2 * expY + shock_i

        if pos == 0:
            C_outcome[pos:pos+3] = np.dot(A_inv,B)
        else:
            C_outcome[pos:pos+3] = np.dot(A_inv,B)-C_outcome[0:3]
        
    equations = C_outcome - np.array(coef)
    return equations

init = [0] * 9

[msv_coefficients, inf, ier, msg] = fsolve(statefunc, init, full_output = True)
if ier != 0:
    print(msg)

msv_constants = msv_coefficients[0:3]
msv_shock_Y_coefs = msv_coefficients[3:6]
msv_shock_Pi_coefs = msv_coefficients[6:9]


print('The constants in the msv solution are: ', msv_constants)
print('The coefficients on the output gap shock are: ', msv_shock_Y_coefs)
print('The coefficients on the interest rate shock are: ', msv_shock_Y_coefs)

# Import relevant libraries

from itertools import zip_longest
import numpy as np
import operator as op
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from sys import exit
# import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from collections import Counter
# from scipy.signal import savgol_filter
from scipy.optimize import minimize

# Say, "the default sans-serif font is COMIC SANS"
mpl.rcParams['font.sans-serif'] = "Comic Sans MS"
# Then, "ALWAYS use sans-serif fonts"
mpl.rcParams['font.family'] = "Arial"

def one(x):
    return np.ones_like(x)

def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def tan(x):
    return np.tan(x)

def exp(x):
    return np.exp(x)

def neg_exp(x):
    return np.exp(-x)

def linear(x):
    return x

def square(x):
    return x ** 2

def cube(x):
    return x ** 3

def quart(x):
    return x ** 4

def log(x):
    return np.log(x)

def f(i):
    switcher_f = {
        0: one,
        1: sin,
        2: cos,
        3: tan,
        4: exp,
        5: linear,
        6: square,
        7: cube,
        8: quart,
        9: neg_exp,
        10: log
    }
    fform = switcher_f.get(i, "Invalid")
    return fform

def o(j):
    switcher_o = {
        0: op.mul,
        1: op.truediv
    }
    oform = switcher_o.get(j, "Invalid")
    return oform

def bvx(vx):
    if vx == 0:
        return 'Revenue'
    elif vx == 1:
        return 'Net income'
    elif vx == 2:
        return 'Operating income'
    elif vx == 3:
        return 'Gross margin'
    elif vx == 4:
        return 'P/E Ratio'
    elif vx == 5:
        return 'P/S Ratio'
    elif vx == 6:
        return 'Current Ratio'
    elif vx == 7:
        return 'Cash and cash equivalents'
    elif vx == 8:
        return 'Net Cash from Operations'
    elif vx == 9:
        return 'Research and development'
    elif vx == 10:
        return 'EPS'

 
def bfxvar(fx,var):
    switcher_name = {
        0: "1",
        1: "np.sin("+bvx(var)+")",
        2: "np.cos("+bvx(var)+")",
        3: "np.tan("+bvx(var)+")",
        4: "np.exp("+bvx(var)+")",
        5: "("+bvx(var)+")",
        6: "(("+bvx(var)+")**2)",
        7: "(("+bvx(var)+")**3)",
        8: "(("+bvx(var)+")**4)",
        9: "np.exp(-"+bvx(var)+")",
        10: "np.log("+bvx(var)+")"
    }
    beautiful_name = switcher_name.get(fx, "Invalid")
    return beautiful_name

def bop(op):
    switcher_op = {
        0: "*",
        1: "/"
    }
    beautiful_op = switcher_op.get(op, "Invalid")
    return beautiful_op

def beautifyoutputwithvar(fx,var,op,par,addConst=False):
    fList = []
    for i in range(len(fx)):
        # fList.append(str(round(par[i+int(addConst)], 10)));fList.append("*")
        fList.append(str(round(par[i], 10)));fList.append("*")
        for j,k,l in zip_longest(enumerate(fx[i]),enumerate(var[i]),enumerate(op[i])):
            if l is not None:
                fList.append(bfxvar(int(fx[i][j[0]]),int(var[i][k[0]]))+bop(int(op[i][l[0]])))
            if l is None:
                fList.append(bfxvar(int(fx[i][j[0]]),int(var[i][k[0]])))
        if i < len(fx)-1:
            fList.append(" + ")
    if addConst == True:
        fList.append(" + " + str(round(par[0], 20)))
    else:
        pass
    return ''.join(fList)

def beautify_single_feature(fx, var, op):
    feature_str = []
    for i in range(len(fx)):
        feature_str.append(bfxvar(fx[i], var[i]))
        if i == len(op):
            pass
        else:
            feature_str.append(bop(op[i]))
    return ''.join(feature_str)

" Import data as Python pandas DataFrame "

#########################################################################################################

with open('TRAIN.npy', 'rb') as TRAIN_FILE:
    X_ = np.load(TRAIN_FILE, allow_pickle=True)
    Y_ = np.load(TRAIN_FILE, allow_pickle=True)

# Obtain mean & standard deviation of the inputs/features/indepedents
x_train_temp_MEAN = np.mean(X_, axis=0)
x_train_temp_STD = np.std(X_, axis=0)

# Normalize the inputs/features/indepedents
x_train_temp = (X_ - x_train_temp_MEAN) / x_train_temp_STD

X_train = x_train_temp
Y_train = Y_.reshape(-1, 1)

print("Shape of X_train:", X_train.shape)
print("Shape of Y_train:", Y_train.shape)

# Now for test data

with open('TEST.npy', 'rb') as TEST_FILE:
    _X_test = np.load(TEST_FILE, allow_pickle=True)
    _Y_test = np.load(TEST_FILE, allow_pickle=True)

# Normalize the inputs/features/indepedents
x_test_temp = (_X_test - x_train_temp_MEAN) / x_train_temp_STD

X_test = x_test_temp
Y_test = _Y_test.reshape(-1, 1)

print("Shape of X_test:", X_test.shape)
print("Shape of Y_test:", Y_test.shape)

X = np.vstack((X_train, X_test)).T
Y = np.vstack((Y_train, Y_test))

print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

#########################################################################################################

" Options to choose from "

if X.ndim == 1:
    siso = True
elif X.ndim > 1:
    siso = False

fOps = [0, 4, 5, 6, 7, 9]
if siso == False:
    vOps = list(range(0, X.shape[0]))
elif siso == True:
    vOps = [0]
oOps = [0, 1]          # 0 –> multiplication, 1 –> division

" Mention the parameters of MISO GFEST "

pop = 20; m = 12; k = 3; maxGen = 1000; zMSE = []#; alpha = 0.05;  l1_ratio = 0.02; max_iter = 1000

" Generate Zeroth population "

zFs = np.random.choice(fOps,size=pop*m*k).reshape(pop,m,k)
vFs = np.random.choice(vOps,size=pop*m*k).reshape(pop,m,k)
if k > 1:
    oFs = np.random.choice(oOps,size=pop*m*(k-1)).reshape(pop,m,k-1)
elif k == 1:
    oFs = np.random.choice(oOps,size=pop*m*1).reshape(pop,m,1)

def funcletEval(funclet,varlet,oplet):
    """
    This function evaluates the value of the funclet.
    :return: Value of the funclet.
    """
    if siso == False:
        funcletVal = f(funclet[0])(X[varlet[0], :])
    elif siso == True:
        funcletVal = f(funclet[0])(X)

    if siso == False:
        if k > 1:
            for i in range(k - 1):
                funcletVal = o(oplet[i])(funcletVal, f(funclet[i + 1])(X[varlet[i + 1], :]))
            return funcletVal
        elif k == 1:
            return funcletVal
    elif siso == True:
        if k > 1:
            for i in range(k - 1):
                funcletVal = o(oplet[i])(funcletVal, f(funclet[i + 1])(X))
            return funcletVal
        elif k == 1:
            return funcletVal

def regressorMatrix(func,var,op):
    if m == 1:
        regMatrix = funcletEval(func[0],var[0],op[0]).T
    if m > 1:
        temp = funcletEval(func[0],var[0],op[0])
        for i in range(1,m):
            temp = np.vstack((temp,funcletEval(func[i],var[i],op[i])))
        regMatrix = temp.T
    return regMatrix

zMSE = []

for individualIndex in range(pop):

    # # Example for handling train and test regressor matrix
    # zeroRM_train = regressorMatrix(zFs[individualIndex], vFs[individualIndex], oFs[individualIndex])
    # zeroRM_test = regressorMatrix(zFs[individualIndex], vFs[individualIndex], oFs[individualIndex])

    zeroRM = regressorMatrix(zFs[individualIndex], vFs[individualIndex], oFs[individualIndex])

    zeroRM_train = zeroRM[:46,:]; _y_train_ = Y[:46]
    zeroRM_test = zeroRM[46:,:]; _y_test_ = Y[46:]

    # Fitting model on the training data
    nregr_OLS = LinearRegression(fit_intercept=False).fit(zeroRM_train, _y_train_)
    
    # Making predictions on both training and testing data
    Y_train_pred = nregr_OLS.predict(zeroRM_train)
    Y_test_pred = nregr_OLS.predict(zeroRM_test)

    Y_pred = nregr_OLS.predict(zeroRM)

    # Calculating metrics for both training and testing
    train_r2 = r2_score(_y_train_, Y_train_pred)
    train_mae = mean_absolute_error(_y_train_, Y_train_pred)
    test_r2 = r2_score(_y_test_, Y_test_pred)
    test_mae = mean_absolute_error(_y_test_, Y_test_pred)

    # Print the training and testing metrics
    print(f"Individual {individualIndex + 1}")
    print("Training R² =", train_r2)
    print("Training MAE =", train_mae)
    print("Testing R² =", test_r2)
    print("Testing MAE =", test_mae)
    print()

    zMSE.append(test_mae)


zeroMSE = np.asarray(zMSE); sortedidx = np.argsort(zeroMSE); sortedZeroMSE = zeroMSE[sortedidx[0::]]
sortedzFs = zFs[sortedidx[0::]]; sortedvFs = vFs[sortedidx[0::]]; sortedoFs = oFs[sortedidx[0::]]

sortedPop = np.copy(sortedzFs); sortedVar = np.copy(sortedvFs); sortedOp = np.copy(sortedoFs)

" Future Generations "

nSel = 3; nFmut = 3; nVmut = 3; nOmut = 2; nCrsvr = pop - nSel - nFmut

for gen in range(maxGen):

    # print(); print("Generation " + str(gen + 1)); print()

    sortedPopCopy = np.copy(sortedPop); sortedVarCopy = np.copy(sortedVar); sortedOpCopy = np.copy(sortedOp)

    # Perform Function Mutation

    for iFmut in range(nFmut):
        tempFunc = sortedPop[rd.randint(0, nSel - 1)]
        tempFunc[rd.randint(0, m - 1)][rd.randint(0, k - 1)] = rd.choice(fOps)
        sortedPopCopy[nSel + iFmut] = tempFunc

    # Perform Variable Mutation

    for iVmut in range(nVmut):
        tempVar = sortedVar[rd.randint(0, nSel - 1)]
        tempVar[rd.randint(0, m - 1)][rd.randint(0, k - 1)] = rd.choice(vOps)
        sortedVarCopy[nSel + iVmut] = tempVar

    # Perform Operator Mutation

    if k > 1:
        for iOmut in range(nOmut):
            tempOp = sortedOp[rd.randint(0, nSel - 1)]
            tempOp[rd.randint(0, m - 1)][rd.randint(0, k - 2)] = rd.choice(oOps)
            sortedOpCopy[nSel + iOmut] = tempOp

    # Perform Crossover

    for iCrsvr in range(nCrsvr):
        crsvrIndex = rd.randint(0, k - 1)
        crossedOver = np.concatenate((sortedPop[rd.randint(0,pop-1)][rd.randint(0,m-1)][0:crsvrIndex],
                                      sortedPop[rd.randint(0,pop-1)][rd.randint(0,m-1)][crsvrIndex:]))
        sortedPopCopy[nSel + nFmut + iCrsvr] = crossedOver

    # Selection is taken care of in the substitution

    nextPop = sortedPopCopy; nextVar = sortedVarCopy; nextOp = sortedOpCopy

    # Evaluate n-th Population

    nMSE = []

    # print(); print("Generation " + str(gen + 1)); print()

    if (gen + 1) % 100 == 0:
        print(); print("Generation " + str(gen + 1)); print()
    else:
        pass

    for indiIndex in range(pop):

        nRM = regressorMatrix(nextPop[indiIndex], nextVar[indiIndex], nextOp[indiIndex])

        nRM_train = nRM[:46,:]; _y_train_ = Y[:46]
        nRM_test = nRM[46:,:]; _y_test_ = Y[46:]

        # Fitting model on the training data
        nregr_OLS = LinearRegression(fit_intercept=False).fit(nRM_train, _y_train_)
        
        # Making predictions on both training and testing data
        Y_train_pred = nregr_OLS.predict(nRM_train)
        Y_test_pred = nregr_OLS.predict(nRM_test)

        Y_pred = nregr_OLS.predict(nRM)

        intercept = nregr_OLS.intercept_
        regression_params = np.ravel(nregr_OLS.coef_)

        ############################ NO CUSTOM LOSS ############################

        if (gen + 1) % 100 == 0 and indiIndex < nSel:
            print("Individual " + str(indiIndex + 1) + ": \n")
            print("R² = %f" % r2_score(Y, Y_pred))
            print("MAE = %f" % mean_absolute_error(Y, Y_pred))
            nCoeff_OLS_skl = np.hstack((intercept, regression_params))
            print(beautifyoutputwithvar(nextPop[indiIndex], nextVar[indiIndex], nextOp[indiIndex], nCoeff_OLS_skl, False))
            print()
        else:
            pass

        nMSE.append(mean_absolute_error(Y, Y_pred))

    nGenMSE = np.asarray(nMSE); sortedidx = np.argsort(nGenMSE); sortednMSE = nGenMSE[sortedidx[0::]]
    sortedPop = nextPop[sortedidx[0::]]; sortedVar = nextVar[sortedidx[0::]]; sortedOp = nextOp[sortedidx[0::]]

################################################################################################

print('\n################################################################################################')

nRM_ = regressorMatrix(sortedPop[0], sortedVar[0], sortedOp[0])

nRM_train = nRM_[:46,:]; _y_train_ = Y[:46]
nRM_test = nRM_[46:,:]; _y_test_ = Y[46:]

# Fitting model on the training data
nregr_OLS = LinearRegression(fit_intercept=False).fit(nRM_train, _y_train_)

# Making predictions on both training and testing data
Y_train_pred = nregr_OLS.predict(nRM_train)
Y_test_pred = nregr_OLS.predict(nRM_test)

Y_pred = nregr_OLS.predict(nRM_)

intercept = nregr_OLS.intercept_
regression_params = np.ravel(nregr_OLS.coef_)

print("R² = %f" % r2_score(Y, Y_pred))
print("MAE = %f" % mean_absolute_error(Y, Y_pred))

nCoeff_OLS_skl = np.hstack((intercept, regression_params))

print(beautifyoutputwithvar(sortedPop[0], sortedVar[0], sortedOp[0], nCoeff_OLS_skl, False))
print()

# # Create just a figure and only one subplot
fig, ax = plt.subplots(1, 3, figsize=(18,6), sharey=True)

p1 = max(max(Y_pred), max(Y))
p2 = min(min(Y_pred), min(Y))

ax[0].scatter(Y, Y_pred, c='crimson')
ax[0].plot([p1, p2], [p1, p2], 'b-')
ax[0].set_ylabel('Predictions', fontsize=14)
ax[0].set_xlabel('True Values', fontsize=14)
ax[0].set_title("Train + Test | R² = %.2f | MAE = %.3f" % (r2_score(
             Y, Y_pred), mean_absolute_error(Y, Y_pred)))

ax[1].scatter(_y_train_, Y_train_pred, c='crimson')
ax[1].plot([p1, p2], [p1, p2], 'b-')
ax[1].set_ylabel('Predictions', fontsize=14)
ax[1].set_xlabel('True Values', fontsize=14)
ax[1].set_title("Train | R² = %.2f | MAE = %.3f" % (r2_score(
             _y_train_, Y_train_pred), mean_absolute_error(_y_train_, Y_train_pred)))

ax[2].scatter(_y_test_, Y_test_pred, c='crimson')
ax[2].plot([p1, p2], [p1, p2], 'b-')
ax[2].set_ylabel('Predictions', fontsize=14)
ax[2].set_xlabel('True Values', fontsize=14)
ax[2].set_title("Test | R² = %.2f | MAE = %.3f" % (r2_score(
             _y_test_, Y_test_pred), mean_absolute_error(_y_test_, Y_test_pred)))

fig.suptitle(r'company stock price prediction', fontsize=18)
plt.savefig('company stock price prediction' + '.png', dpi=300)


f = open('results' + '.txt', 'w')

f.write('These are the mean of the input variables:\n\n')
f.write('Mean of inputs: ' + str(x_train_temp_MEAN))
f.write('\n\nThese are the standard deviations of the ' + str(X.shape[1]) + ' input variables:\n\n')
f.write('Standard deviation of inputs: ' + str(x_train_temp_STD))
f.write('\n\nPlease note that the variables used in the model are normalized: normalized var = (input var - mean of input var)/(std dev of input var)\n\n')
f.write('stock_price = ' + beautifyoutputwithvar(sortedPop[0], sortedVar[0], sortedOp[0], nCoeff_OLS_skl, True))

f.write('\n\nIntercept: ' + str(nCoeff_OLS_skl[0]))
f.write('\n\nCoefficients: ' + str(nCoeff_OLS_skl[1:]))
f.write('\n\nTrain + Test R²: ' + str(r2_score(Y, Y_pred)) + ' | Train + Test MAE: ' + str(mean_absolute_error(Y, Y_pred)))


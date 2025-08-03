import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import t
from arch.univariate.distribution import SkewStudent
from scipy.integrate import quad
from scipy.special import beta
from scipy.optimize import minimize
from scipy.stats import uniform
import pandas as pd



"""  PARAMETRI DI BASE """
BurnIn = 10**4 
PostBurnIn = 2500
N = BurnIn + PostBurnIn
K = 1000 # lag massimo --> da inf viene troncato
Presample = np.zeros(K) # valori fuori dal campione vengono inizializzati a 0
B = 40 # nr di replicazioni della simulazione per poter poi confrontare i risultati
ALPHA = 0.025
NR_CANDID = 50
NR_PASS_CHECK = 5
DELTA = 0.005


# definizioni indicatrici negativa e positiva
def I_val(x, y):
    return 1 if x > y else 0

"""  SPECIFICAZIONI CAViaR -> ritornano il VaR al tempo t secondo i vari modelli """


#   GEN - - - GENERAZIONE DI v_t SECONDO LA SPECIFICAZIONE SELEZIONATA
def gen_CSASAV(t, v_z, YY_t):
    b0, b1, b2, b3 = 0.05*abs(v_z), 0.8, 0.15*abs(v_z), 0.9

    q = 1 / b3 - 1
    p = q * b1 / (1 - b1)

    ks = np.arange(1, K)
    YY_ts = np.abs(YY_t[t - ks])  

    cost = ((1 - b1) * beta(p, q))
    arg = beta(p - 1 + ks, q + 1) / cost
    
    TotalSum = np.dot(arg, YY_ts)

    v_t = - (b0 / (1 - b1) + b2 * TotalSum)

    return v_t


def gen_CSAAS(t, v_z, YY_t):
    b0, b1, b2m, b2p, b3 = 0.05*abs(v_z), 0.8, 0.25*abs(v_z), 0.05*abs(v_z), 0.9

    q = 1 / b3 - 1
    p = q * b1 / (1 - b1)

    ks = np.arange(1, K)
    YY_ts = np.abs(YY_t[t - ks])

    I_m = (YY_t[t - ks] < 0).astype(float)
    I_p = (YY_t[t - ks] > 0).astype(float)

    cost = ((1 - b1) * beta(p, q))
    arg = beta(p - 1 + ks, q + 1) / cost

    YY_ts_I = (b2m * I_m + b2p * I_p) * YY_ts

    TotalSum = np.dot(arg, YY_ts_I)

    v_t = - ((b0 / (1 - b1)) + TotalSum)

    return v_t


def gen_CSAIG(t, v_z, YY_t):
    b0, b1, b2, b3 = 0.05*v_z**2, 0.8, 0.15*v_z**2, 0.9

    q = 1 / b3 - 1
    p = q * b1 / (1 - b1)

    ks = np.arange(1, K)
    YY_ts = np.array(YY_t[t - ks]**2)  

    cost = ((1 - b1) * beta(p, q))
    arg = beta(p - 1 + ks, q + 1) / cost
    
    TotalSum = np.dot(arg, YY_ts)

    v_t = - np.sqrt(b0 / (1 - b1) + b2 * TotalSum)

    return v_t


def gen_SAV(t, v_z, YY_t):
    b0, b1, b2 = 0.05*abs(v_z), 0.8, 0.15*abs(v_z)

    ks = np.arange(1, K)
    YY_ts = np.abs(YY_t[t - ks])  
    b1s = b1**(ks-1)    
    
    TotalSum = np.dot(b1s, YY_ts)

    v_t = - (b0 / (1 - b1) + b2 * TotalSum)
    
    return v_t
    

def gen_AS(t, v_z, YY_t):
    b0, b1, b2m, b2p = 0.05*abs(v_z), 0.8, 0.25*abs(v_z), 0.05*abs(v_z)

    ks = np.arange(1, K)
    YY_ts = np.abs(YY_t[t - ks])
    b1s = b1**(ks-1) 

    I_m = (YY_t[t - ks] < 0).astype(float)
    I_p = (YY_t[t - ks] > 0).astype(float)

    YY_ts_I = (b2m * I_m + b2p * I_p) * YY_ts

    TotalSum = np.dot(b1s, YY_ts_I)

    v_t = - ((b0 / (1 - b1)) + TotalSum)

    return v_t


def gen_IG(t, v_z, YY_t):
    b0, b1, b2 = 0.05*v_z**2, 0.8, 0.15*v_z**2

    ks = np.arange(1, K)
    YY_ts = np.array(YY_t[t - ks]**2)  
    b1s = b1**(ks-1)    
    
    TotalSum = np.dot(b1s, YY_ts)

    v_t = - np.sqrt(b0 / (1 - b1) + b2 * TotalSum)

    return v_t


def ChooseSpecificGen(specif, z_t, v_z):
    # concateno una sola volta
    YY_t = np.concatenate((Presample, z_t))
    V_t = np.zeros(N)

    mapping = {
        "SAV": gen_SAV,
        "AS":  gen_AS,
        "IG":  gen_IG,
        "CSA-SAV": gen_CSASAV,
        "CSA-AS":  gen_CSAAS,
        "CSA-IG":  gen_CSAIG,
    }

    try:
        func = mapping[specif]
    except KeyError:
        raise NotImplementedError(f"Specificazione {specif} non implementata")

    for t in range(len(z_t)):
        V_t[t] = func(t=t+K, v_z=v_z, YY_t=YY_t)

    return V_t



""" Funzioni che in base alla densità scelta generano z_t e calcolano v_z, e_z """
def Norm(N, seed, alpha):
    z_t = norm.rvs(size=N, random_state=seed)
    v_z = norm.ppf(alpha)
    e_z = truncnorm(a=-np.inf, b=v_z, loc=0, scale=1).mean() 
    return z_t, v_z, e_z


def mean_truncated_t(df, a, b):
    Z = t.cdf(b, df) - t.cdf(a, df) # costante di normalizzazione
    integrand = lambda x: x * t.pdf(x, df) # x*f(x)
    mean, _ = quad(integrand, a, b)
    return mean / Z


def t_Stud(N, df, seed, alpha):
    z_t = t.rvs(df=df, size=N, random_state=seed)
    v_z = t.ppf(q=alpha, df=df)
    e_z = mean_truncated_t(df=df, a=-np.inf, b=t.ppf(q=alpha, df=df))
    return z_t, v_z, e_z



""" Struttura generale per il processo generatore dei dati """

def CSA_CAViaR_FZ(pdf, alpha, specif, seed):
    # 1 - Generare z_t
    # 2 - Calcolare v_z, e_z
    if pdf == "Norm":
        z_t, v_z, e_z = Norm(N=N, seed=seed, alpha=alpha)
    elif pdf == "t-Stud":
        z_t, v_z, e_z = t_Stud(N=N, df=5, seed=seed, alpha=alpha)
    else:
        raise NotImplementedError("Distribuzione non riconosciuta.")

    # 3 - Calcolare gamma
    gamma = (e_z/v_z) - 1

    # 4 - Calcolare v_t tramite una specificazione
    V_t = ChooseSpecificGen(specif=specif, z_t=z_t, v_z=v_z)

    # 5 - Calcolare e_t, a partire da v_t
    E_t = (1 + gamma) * V_t

    # 6 - Calcolare y_t
    Y_t = (V_t / v_z) * z_t

    return Y_t, V_t, E_t





""" Stima dei parametri del modello """
# def L_FZ0(y_t, v_t, e_t, alpha):
    # if e_t >=0:
    #     return np.nan
    # loss = - 1 / (alpha*e_t) * I_val(x=v_t, y=y_t) * (v_t-y_t) + v_t / e_t + np.log(-e_t) - 1
    # return loss


def L_FZ0(y_t, v_t, e_t, alpha):
    if e_t >= -0.025 or v_t >= -0.025 or np.isinf(e_t) or np.isinf(v_t) or np.isnan(e_t) or np.isnan(v_t) or np.isnan(y_t) or np.isinf(y_t):
        return np.inf

    try:
        term1 = -1 / (alpha * e_t)
        term2 = I_val(x=v_t, y=y_t) * (v_t - y_t)
        term3 = v_t / e_t
        term4 = np.log(-e_t)
        loss = term1 * term2 + term3 + term4 - 1
        return loss
    except (FloatingPointError, ZeroDivisionError, ValueError):
        return np.inf



#   EST - - - STIMA DI v_t, e_t SECONDO LA SPECIFICAZIONE SELEZIONATA E I PARAMETRI theta DA STIMARE
def est_CSASAV(YY_t, t, theta):
    b0, b1, b2, b3, gamma = theta
    q = 1 / b3 - 1
    p = q * b1 / (1 - b1)
    
    ks = np.arange(1, K)
    YY_ts = np.abs(YY_t[t - ks])  

    cost = ((1 - b1) * beta(p, q))
    arg = beta(p - 1 + ks, q + 1) / cost
    
    TotalSum = np.dot(arg, YY_ts)

    v_t = - (b0 / (1 - b1) + b2 * TotalSum)
    
    e_t = (1 + gamma) * v_t

    return v_t, e_t



def est_CSAAS(YY_t, t, theta):
    b0, b1, b2m, b2p, b3, gamma = theta
    q = 1 / b3 - 1
    p = q * b1 / (1 - b1)

    ks = np.arange(1, K)
    YY_ts = np.abs(YY_t[t - ks])

    I_m = (YY_t[t - ks] < 0).astype(float)
    I_p = (YY_t[t - ks] > 0).astype(float)

    cost = ((1 - b1) * beta(p, q))
    arg = beta(p - 1 + ks, q + 1) / cost

    Z_ts_I = (b2m * I_m + b2p * I_p) * YY_ts

    TotalSum = np.dot(arg, Z_ts_I)

    v_t = - ((b0 / (1 - b1)) + TotalSum)
    e_t = (1 + gamma) * v_t

    return v_t, e_t



def est_CSAIG(YY_t, t, theta):
    b0, b1, b2, b3, gamma = theta
    q = 1 / b3 - 1
    p = q * b1 / (1 - b1)

    ks = np.arange(1, K)
    YY_ts = np.array(YY_t[t - ks]**2)  

    cost = ((1 - b1) * beta(p, q))
    arg = beta(p - 1 + ks, q + 1) / cost
    
    TotalSum = np.dot(arg, YY_ts)

    v_t = - np.sqrt(b0 / (1 - b1) + b2 * TotalSum)
    e_t = (1 + gamma) * v_t

    return v_t, e_t



def est_SAV(YY_t, t, theta):
    b0, b1, b2, gamma = theta
    ks = np.arange(1, K)
    YY_ts = np.abs(YY_t[t - ks])  
    b1s = b1**(ks-1)    
    
    TotalSum = np.dot(b1s, YY_ts)

    v_t = - (b0 / (1 - b1) + b2 * TotalSum)
    e_t = (1 + gamma) * v_t

    return v_t, e_t
    


def est_AS(YY_t, t, theta):
    b0, b1, b2m, b2p, gamma = theta
    ks = np.arange(1, K)
    YY_ts = np.abs(YY_t[t - ks])
    b1s = b1**(ks-1) 

    I_m = (YY_t[t - ks] < 0).astype(float)
    I_p = (YY_t[t - ks] > 0).astype(float)

    Z_ts_I = (b2m * I_m + b2p * I_p) * YY_ts

    TotalSum = np.dot(b1s, Z_ts_I)

    v_t = - ((b0 / (1 - b1)) + TotalSum)
    e_t = (1 + gamma) * v_t

    return v_t, e_t



def est_IG(YY_t, t, theta):
    b0, b1, b2, gamma = theta
    ks = np.arange(1, K)
    Z_ts = np.array(YY_t[t - ks]**2)  
    b1s = b1**(ks-1)    
    
    TotalSum = np.dot(b1s, Z_ts)

    v_t = - np.sqrt(b0 / (1 - b1) + b2 * TotalSum)
    e_t = (1 + gamma) * v_t

    return v_t, e_t



def ChooseSpecificEst(YY_t, t, specif, theta):
    mapping = {
        "SAV": est_SAV,
        "AS":  est_AS,
        "IG":  est_IG,
        "CSA-SAV": est_CSASAV,
        "CSA-AS":  est_CSAAS,
        "CSA-IG":  est_CSAIG,
    }

    try:
        func = mapping[specif]
    except KeyError:
        raise NotImplementedError(f"Specificazione {specif} non implementata")

    v_t, e_t = func(YY_t=YY_t, t=t, theta=theta)

    return v_t, e_t



# loss "finale" da minimizzare
def L_T(theta, Y_t, alpha, specif, bounds):
    YY_t = np.concatenate((Presample, Y_t))

    losses = []
    # for t in range(len(Y_t)):
    for t in range(K + BurnIn, K + N):
        v_t, e_t = ChooseSpecificEst(YY_t=YY_t, t=t, specif=specif, theta=theta)
        loss = L_FZ0(YY_t[t], v_t=v_t, e_t=e_t, alpha=alpha)
        losses.append(loss)
    
    if any(np.isnan(losses)):
        return np.inf   

    return np.mean(losses) 



def boundsSpecific(specif):
    if specif == "CSA-SAV":
        bounds = [(0, 2), (0, 1), (0, 2), (0, 1), (0, 1)] # b0, b1, b2, b3, gamma
        
    elif specif == "CSA-AS":
        bounds = [(0, 2), (0, 1), (0, 2), (0, 2), (0, 1), (0, 1)] # b0, b1, b2m, b2p, b3, gamma
        
    elif specif == "CSA-IG":
        bounds = [(0, 2), (0, 1), (0, 2), (0, 1), (0, 1)] # b0, b1, b2, b3, gamma

    elif specif == "SAV":
        bounds = [(0, 1), (0, 1), (0, 2), (0, 1)] # b0, b1, b2, gamma
    
    elif specif == "AS":
        bounds = [(0, 2), (0, 1), (0, 2), (0, 2), (0, 1)] # b0, b1, b2m, b2p, gamma

    elif specif == "IG":
        bounds = [(0, 2), (0, 1), (0, 2), (0, 1)] # b0, b1, b2, gamma
    else:
        raise NotImplementedError(f"Specificazione {specif} non implementata")
    
    return bounds



# performa il metodo Quasi-Newton
def QuasiNewton(theta0, Y_t, alpha, specif, bounds):
    res = minimize(L_T, theta0, args=(Y_t, alpha, specif, bounds), method="L-BFGS-B", bounds=bounds) 
    return res.x # theta_hat





#   PAR - - - PARAMETRI ORIGINALI E STARTING POINT UNIF
def par_CSASAV(v_z, e_z):
    gamma = (e_z/v_z) - 1
    b0, b1, b2, b3 = 0.05*abs(v_z), 0.8, 0.15*abs(v_z), 0.9

    params = np.array([b0, b1, b2, b3, gamma])
    Unif = np.column_stack((params - DELTA, np.full_like(params, 2 * DELTA)))

    return params, Unif


def par_CSAAS(v_z, e_z):
    gamma = (e_z/v_z) - 1
    b0, b1, b2m, b2p, b3 = 0.05*abs(v_z), 0.8, 0.25*abs(v_z), 0.05*abs(v_z), 0.9

    params = np.array([b0, b1, b2m, b2p, b3, gamma])
    Unif = np.column_stack((params - DELTA, np.full_like(params, 2 * DELTA)))

    return params, Unif


def par_CSAIG(v_z, e_z):
    gamma = (e_z/v_z) - 1
    b0, b1, b2, b3 = 0.05*v_z**2, 0.8, 0.15*v_z**2, 0.9

    params = np.array([b0, b1, b2, b3, gamma])
    Unif = np.column_stack((params - DELTA, np.full_like(params, 2 * DELTA)))

    return params, Unif


def par_SAV(v_z, e_z):
    gamma = (e_z/v_z) - 1
    b0, b1, b2 = 0.05*abs(v_z), 0.8, 0.15*abs(v_z)
    
    params = np.array([b0, b1, b2, gamma])
    Unif = np.column_stack((params - DELTA, np.full_like(params, 2 * DELTA)))

    return params, Unif
    

def par_AS(v_z, e_z):
    gamma = (e_z/v_z) - 1
    b0, b1, b2m, b2p = 0.05*abs(v_z), 0.8, 0.25*abs(v_z), 0.05*abs(v_z) 
    
    params = np.array([b0, b1, b2m, b2p, gamma])
    Unif = np.column_stack((params - DELTA, np.full_like(params, 2 * DELTA)))

    return params, Unif


def par_IG(v_z, e_z):
    gamma = (e_z/v_z) - 1
    b0, b1, b2 = 0.05*v_z**2, 0.8, 0.15*v_z**2
    
    params = np.array([b0, b1, b2, gamma])
    Unif = np.column_stack((params - DELTA, np.full_like(params, 2 * DELTA)))

    return params, Unif



def ChooseSpecificPar(pdf, specif, alpha, seed):
    if pdf == "Norm":
        z_t, v_z, e_z = Norm(N=N, seed=seed, alpha=alpha)
    elif pdf == "t-Stud":
        z_t, v_z, e_z = t_Stud(N=N, df=5, seed=seed, alpha=alpha)
    else:
        raise NotImplementedError("Distribuzione non riconosciuta.")

    mapping = {
        "SAV": par_SAV,
        "AS":  par_AS,
        "IG":  par_IG,
        "CSA-SAV": par_CSASAV,
        "CSA-AS":  par_CSAAS,
        "CSA-IG":  par_CSAIG,
    }

    try:
        func = mapping[specif]
    except KeyError:
        raise NotImplementedError(f"Specificazione {specif} non implementata")

    return func(v_z=v_z, e_z=e_z)



def ParamEstimation(nr_candid, nr_pass_check, specif, pdf, Y_t, alpha, seed, bounds):
    # 1 - Generare n=nr_candid parametri basandosi su una distrib Unif
    ThetaTrue, Unif = ChooseSpecificPar(pdf=pdf, specif=specif, alpha=alpha, seed=seed)
    Uniform = [uniform(loc=row[0], scale=row[1]) for row in Unif]
    Theta0 = np.array([dist.rvs(size=nr_candid) for dist in Uniform]).T

    # 2 - Selezionare n=nr_pass_check vettori di parametri che minimizzano L_FZ0 
    loss_values1 = np.zeros(nr_candid)
    for i in range(nr_candid):
        loss_values1[i] = L_T(theta=Theta0[i, :], Y_t=Y_t, alpha=alpha, specif=specif, bounds=bounds)
        # print(f"Valutato candidato {i}") 

    best_index1 = np.argsort(loss_values1)[:nr_pass_check] # trovo gli indici dei migliori
    BestTheta0 = Theta0[best_index1, :]

    # 3 - Utilizzare ciascuno di questi nr_pass_check vettori come punto di partenza per l'algoritmo Quasi-Newton
    BestThetaQN = np.zeros((nr_pass_check, np.shape(BestTheta0)[1]))
    for i in range(nr_pass_check):
        BestThetaQN[i, :] = QuasiNewton(theta0=BestTheta0[i, :], Y_t=Y_t, alpha=alpha, specif=specif, bounds=bounds)
        # print(f"Trovo tramite Q-N l'{i}-esimo theta_hat_QN ")

    # 4 - Ottengo il vettore di parametri migliore come quello che tra i nr_pass_check minimizza L_FZ0
    loss_values2 = np.zeros(nr_pass_check)
    for i in range(nr_pass_check):
        loss_values2[i] = L_T(theta=BestThetaQN[i, :], Y_t=Y_t, alpha=alpha, specif=specif, bounds=bounds)

    best_index2 = np.argsort(loss_values2)[0] # trova l'indice del migliore
    ThetaQN = BestThetaQN[best_index2, :]
    print(f"{specif} -- {ThetaQN}")

    return ThetaQN



def tab_CSASAV(ThetaTrue, BThetaQN):
    param = ["b0", "b1", "b2", "b3", "gamma"]
    meanBThetaQN = np.mean(BThetaQN, axis=0)
    sdBThetaQN = np.std(BThetaQN, ddof=1, axis=0)
    rmseBThetaQN = np.sqrt(((BThetaQN - ThetaTrue) ** 2).mean(axis=0))
    return param, meanBThetaQN, sdBThetaQN, rmseBThetaQN



def tab_CSAAS(ThetaTrue, BThetaQN):
    param = ["b0", "b1", "b2m", "b2p", "b3", "gamma"]
    meanBThetaQN = np.mean(BThetaQN, axis=0)
    sdBThetaQN = np.std(BThetaQN, ddof=1, axis=0)
    rmseBThetaQN = np.sqrt(((BThetaQN - ThetaTrue) ** 2).mean(axis=0))
    return param, meanBThetaQN, sdBThetaQN, rmseBThetaQN



def tab_CSAIG(ThetaTrue, BThetaQN):
    param = ["b0", "b1", "b2", "b3", "gamma"]
    meanBThetaQN = np.mean(BThetaQN, axis=0)
    sdBThetaQN = np.std(BThetaQN, ddof=1, axis=0)
    rmseBThetaQN = np.sqrt(((BThetaQN - ThetaTrue) ** 2).mean(axis=0))
    return param, meanBThetaQN, sdBThetaQN, rmseBThetaQN



def tab_SAV(ThetaTrue, BThetaQN):
    param = ["b0", "b1", "b2", "gamma"]
    meanBThetaQN = np.mean(BThetaQN, axis=0)
    sdBThetaQN = np.std(BThetaQN, ddof=1, axis=0)
    rmseBThetaQN = np.sqrt(((BThetaQN - ThetaTrue) ** 2).mean(axis=0))
    return param, meanBThetaQN, sdBThetaQN, rmseBThetaQN



def tab_AS(ThetaTrue, BThetaQN):
    param = ["b0", "b1", "b2m", "b2p", "gamma"]
    meanBThetaQN = np.mean(BThetaQN, axis=0)
    sdBThetaQN = np.std(BThetaQN, ddof=1, axis=0)
    rmseBThetaQN = np.sqrt(((BThetaQN - ThetaTrue) ** 2).mean(axis=0))
    return param, meanBThetaQN, sdBThetaQN, rmseBThetaQN



def tab_IG(ThetaTrue, BThetaQN):
    param = ["b0", "b1", "b2", "gamma"]
    meanBThetaQN = np.mean(BThetaQN, axis=0)
    sdBThetaQN = np.std(BThetaQN, ddof=1, axis=0)
    rmseBThetaQN = np.sqrt(((BThetaQN - ThetaTrue) ** 2).mean(axis=0))
    return param, meanBThetaQN, sdBThetaQN, rmseBThetaQN



def ChooseSpecificTab(specif, ThetaTrue, BThetaQN):
    mapping = {
        "SAV": tab_SAV,
        "AS":  tab_AS,
        "IG":  tab_IG,
        "CSA-SAV": tab_CSASAV,
        "CSA-AS":  tab_CSAAS,
        "CSA-IG":  tab_CSAIG,
    }

    try:
        func = mapping[specif]
    except KeyError:
        raise NotImplementedError(f"Specificazione {specif} non implementata")

    return func(ThetaTrue=ThetaTrue, BThetaQN=BThetaQN)



def BuildDataFrame(pdf, specif, ThetaTrue, meanBThetaQN, sdBThetaQN, rmseBThetaQN, param):
    df = pd.DataFrame(
        {
            "Specific": specif,
            "Distrib": pdf,
            "Param": param, # lista tipo ["b0", "b1", "b2", "gamma"]
            "True": ThetaTrue, #stessa lunghezza di parametri
            "Mean": meanBThetaQN,
            "SD": sdBThetaQN,
            "RMSE": rmseBThetaQN
        }
    )

    return df



def CompleteOneProcess(pdf, specif, seed, alpha, bounds, nr_candid, nr_pass_check, B=B):
    # 1 - Generare la serie Y_t rispetto alla quale si stimeranno i parametri theta
    Y_t, V_t, E_t = CSA_CAViaR_FZ(pdf=pdf, alpha=alpha, specif=specif, seed=seed)

    # 2 - Ottenere per B volte la stima dei parametri tramite ParamEstimation
    # ( B vettori di parametri theta)
    print(f"{specif} - {pdf} - SIMULAZIONE 0")
    thetaQN = ParamEstimation(nr_candid=nr_candid, nr_pass_check=nr_pass_check, specif=specif, pdf=pdf, Y_t=Y_t, alpha=ALPHA, seed=seed, bounds=bounds)
    n_param = len(thetaQN)

    BThetaQN = np.zeros((B, n_param))
    BThetaQN[0, :] = thetaQN

    for i in range(1, B):
        print(f"{specif} - {pdf} - SIMULAZIONE {i}")
        BThetaQN[i, :] = ParamEstimation(nr_candid=nr_candid, nr_pass_check=nr_pass_check, specif=specif, pdf=pdf, Y_t=Y_t, alpha=ALPHA, seed=seed, bounds=bounds)

    print(BThetaQN)

    # 3 - Media, Deviazione Standard, RMSE rispetto a ThetaTrue + Costruzione tabella
    ThetaTrue, Unif = ChooseSpecificPar(pdf=pdf, specif=specif, alpha=alpha, seed=seed) # il seed non sarà utilizzato

    param, meanBThetaQN, sdBThetaQN, rmseBThetaQN = ChooseSpecificTab(specif=specif, ThetaTrue=ThetaTrue, BThetaQN=BThetaQN)

    df = BuildDataFrame(pdf=pdf, specif=specif, ThetaTrue=ThetaTrue, meanBThetaQN=meanBThetaQN, sdBThetaQN=sdBThetaQN, rmseBThetaQN=rmseBThetaQN, param=param)

    print(f"ThetaMean: {meanBThetaQN}")

    return df


table_configs = [
    # Normale (HT=False) 
    ("Norm",   "SAV",      4224),
    ("Norm",   "AS",       4224),
    ("Norm",   "IG",       4224),

    ("Norm",   "CSA-SAV",  4224),
    ("Norm",   "CSA-AS",   4224),
    ("Norm",   "CSA-IG",   4224),

    # t-Student (HT=True)
    ("t-Stud", "SAV",      4224),
    ("t-Stud", "AS",       4224),
    ("t-Stud", "IG",       4224),

    ("t-Stud", "CSA-SAV",  4224),    
    ("t-Stud", "CSA-AS",   4224),
    ("t-Stud", "CSA-IG",   4224)
]


dfs = []
for (pdf, specif, seed) in table_configs:
    bounds = boundsSpecific(specif=specif)
    df = CompleteOneProcess(pdf=pdf, specif=specif, seed=seed, alpha=ALPHA, bounds=bounds, nr_candid=NR_CANDID, nr_pass_check=NR_PASS_CHECK, B=B)
    dfs.append(df)
    print(f"{specif} - {pdf} completata")

DF = pd.concat(dfs, ignore_index=True)


latex_param = {
    "b0": r"$\beta_0$",
    "b1": r"$\beta_1$",
    "b2": r"$\beta_2$",
    "b2m": r"$\beta_2^-$",
    "b2p": r"$\beta_2^+$",
    "b3": r"$\beta_3$",
    "gamma": r"$\gamma$"
}


DF["Param"] = DF["Param"].map(latex_param)
DF[["Specific", "Distrib"]] = DF[["Specific", "Distrib"]].mask(DF[["Specific", "Distrib"]].duplicated())


latex_table = DF.to_latex(
    index=False,
    escape=False,
    na_rep="",
    column_format="ccccccc",
    float_format="%.3f",
    longtable=True,
    caption="Accuratezza della stima dei parametri",
    label="tab:paramest"
)

with open("ParamEstimation0208.tex", "w") as f:
    f.write(latex_table)



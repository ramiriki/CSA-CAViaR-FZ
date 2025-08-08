import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.special import beta
from scipy.optimize import minimize
from scipy.stats import uniform


df = pd.read_csv(
    "ExchangeRates.csv",
    index_col=0,
    parse_dates=True
)

df.columns = df.columns.str.strip()

Currencies = ["USD/CNY", "EUR/CNY", "100JPY/CNY", "HKD/CNY"]

curr = np.array(df[Currencies[0]])
y_t = 100 * np.diff(np.log(curr))


N = len(y_t)
K = 1000 # lag massimo --> da inf viene troncato # 1
Presample = np.full(K, np.mean(y_t))
B = 40 # nr di replicazioni della simulazione per poter poi confrontare i risultati 
ALPHA = 0.025
NR_CANDID = 50
NR_PASS_CHECK = 5


def I_val(x, y):
    return 1 if x > y else 0


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
    for t in range(K, K + N):
        v_t, e_t = ChooseSpecificEst(YY_t=YY_t, t=t, specif=specif, theta=theta)
        loss = L_FZ0(YY_t[t], v_t=v_t, e_t=e_t, alpha=alpha)
        losses.append(loss)
    
    if any(np.isnan(losses)):
        return np.inf   

    return np.mean(losses) 



def boundsSpecific(specif):
    if specif == "CSA-SAV":
        bounds = [(0, 1), (0, 1), (0, 2), (0, 1), (0, 1)] # b0, b1, b2, b3, gamma
        
    elif specif == "CSA-AS":
        bounds = [(0, 1), (0, 1), (0, 2), (0, 2), (0, 1), (0, 1)] # b0, b1, b2m, b2p, b3, gamma
        
    elif specif == "CSA-IG":
        bounds = [(0, 1), (0, 1), (0, 3), (0, 1), (0, 1)] # b0, b1, b2, b3, gamma

    elif specif == "SAV":
        bounds = [(0, 1), (0, 1), (0, 2), (0, 1)] # b0, b1, b2, gamma
    
    elif specif == "AS":
        bounds = [(0, 1), (0, 1), (0, 2), (0, 2), (0, 1)] # b0, b1, b2m, b2p, gamma

    elif specif == "IG":
        bounds = [(0, 1), (0, 1), (0, 3), (0, 1)] # b0, b1, b2, gamma
    else:
        raise NotImplementedError(f"Specificazione {specif} non implementata")
    
    return bounds



# performa il metodo Quasi-Newton
def QuasiNewton(theta0, Y_t, alpha, specif, bounds):
    res = minimize(L_T, theta0, args=(Y_t, alpha, specif, bounds), method="L-BFGS-B", bounds=bounds) 
    return res.x # theta_hat



#   PAR - - - PARAMETRI ORIGINALI E STARTING POINT UNIF
def StartingPoint(specif):
    if specif == "CSA-SAV":
            theta0 = np.array((uniform(loc=0, scale=0.01), # b0
                            uniform(loc=0.81, scale=0.06), # b1
                            uniform(loc=0.45, scale=0.06), # b2
                            uniform(loc=0.77, scale=0.06), # b3
                            uniform(loc=0.34, scale=0.06))) # gamma 
    
    elif specif == "CSA-AS":
        theta0 = np.array((uniform(loc=0, scale=0.01), # b0
                           uniform(loc=0.82, scale=0.06), # b1
                           uniform(loc=0.48, scale=0.06), # b2m
                           uniform(loc=0.37, scale=0.06), # b2p
                           uniform(loc=0.77, scale=0.06), # b3
                           uniform(loc=0.35, scale=0.06))) # gamma
        
    elif specif == "CSA-IG":
        theta0 = np.array((uniform(loc=0, scale=0.01), # b0
                           uniform(loc=0.76, scale=0.06), # b1
                           uniform(loc=0.93, scale=0.06), # b2
                           uniform(loc=0.63, scale=0.06), # b3
                           uniform(loc=0.34, scale=0.06))) # gamma
    
    elif specif == "SAV":
        theta0 = np.array((uniform(loc=0, scale=0.20), # b0
                           uniform(loc=0.70, scale=0.20), # b1 # 
                           uniform(loc=0.25, scale=0.15), # b2
                           uniform(loc=0.05, scale=0.20))) # gamma 
        
    elif specif == "AS":
        theta0 = np.array((uniform(loc=0, scale=0.20), # b0
                           uniform(loc=0.70, scale=0.20), # b1
                           uniform(loc=0.45, scale=0.20), # b2m
                           uniform(loc=0, scale=0.20), # b2p
                           uniform(loc=0, scale=0.20))) # gamma
    
    elif specif == "IG":
        theta0 = np.array((uniform(loc=0.15, scale=0.20), # b0
                           uniform(loc=0.60, scale=0.30), # b1
                           uniform(loc=0.60, scale=0.30), # b2
                           uniform(loc=0, scale=0.20))) # gamma 
            
    else:
        raise NotImplementedError(f"Specificazione {specif} non implementata")
    
    return theta0



def ParamEstimation(nr_candid, nr_pass_check, specif, Y_t, alpha, bounds):
    # 1 - Generare n=nr_candid parametri basandosi su una distrib Unif
    theta0 = StartingPoint(specif=specif)
    dimTheta = len(theta0)

    # Genera una matrice di dimensione (nr_candid, d)
    Theta0 = np.empty((nr_candid, dimTheta))
    for i in range(dimTheta):
        Theta0[:, i] = theta0[i].rvs(size=nr_candid)

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



def CompleteOneProcess(Y_t, specif, alpha, bounds, nr_candid, nr_pass_check, B=B):
    # 2 - Ottenere per B volte la stima dei parametri tramite ParamEstimation
    # ( B vettori di parametri theta)
    print(f"SIMULAZIONE 0")
    thetaQN = ParamEstimation(nr_candid=nr_candid, nr_pass_check=nr_pass_check, specif=specif, Y_t=Y_t, alpha=alpha, bounds=bounds)
    n_param = len(thetaQN)

    BThetaQN = np.zeros((B, n_param))
    BThetaQN[0, :] = thetaQN

    for i in range(1, B):
        print(f"SIMULAZIONE {i}")
        BThetaQN[i, :] = ParamEstimation(nr_candid=nr_candid, nr_pass_check=nr_pass_check, specif=specif, Y_t=Y_t, alpha=alpha, bounds=bounds)

    theta_hat = np.mean(BThetaQN, axis=0)
    loss_f = L_T(theta=theta_hat, Y_t=Y_t, alpha=alpha, specif=specif, bounds=bounds)

    return np.concatenate((theta_hat, [loss_f]))



table_specif = [
    # Short memory
    # "SAV",
    # "AS",
    # "IG",

    # Long memory
    "CSA-SAV",
    "CSA-AS",
    "CSA-IG"
]


TableParams = {}
for specif in table_specif:
    res = CompleteOneProcess(Y_t=y_t, specif=specif, alpha=ALPHA, bounds=boundsSpecific(specif=specif), nr_candid=NR_CANDID, nr_pass_check=NR_PASS_CHECK, B=B)
    TableParams[specif] = res


# determino numero massimo di parametri
max_len = max(len(v) for v in TableParams.values())

# costruzione indici dinamici
param_labels = [
    r"$\beta_0$",
    r"$\beta_1$",
    r"$\beta_2 \text{ or } \beta_2^{-}$",
    r"$\beta_2^{+}$",
    r"$\beta_3$",
    r"$\gamma$",
    r"$L_{FZ0}$"
]

# mappatura dei parametri disponibili per ogni modello
param_map = {
    "CSA-SAV":   [0, 1, 2,      None, 3, 4, "LFZ0"],  # None → non ha b2+
    "CSA-AS":    [0, 1, 2,      3,    4, 5, "LFZ0"], 
    "CSA-IG":    [0, 1, 2,      None, 3, 4, "LFZ0"]
}

# Costruzione tabella
df_dict = {}
for specif, idx_map in param_map.items():
    vals = []
    model_params = TableParams[specif]
    for idx in idx_map:
        if idx is None:
            vals.append(np.nan)  # Posizione "vuota"
        elif idx == "LFZ0":
            vals.append(L_T(theta=model_params[:-1], Y_t=y_t, alpha=ALPHA, specif=specif, bounds=boundsSpecific(specif)))
        else:
            vals.append(model_params[idx])
    df_dict[specif] = vals

DF = pd.DataFrame(df_dict, index=param_labels)

DFLatex = DF.fillna("").to_latex(
    escape=False,
    column_format="lccc",
    float_format="%.3f",
    longtable=True,
    caption=rf"Stima dei parametri per i vari modelli per la serie dei rendimenti USD/CNY per $\alpha$={ALPHA}",
    label="tab:estimUSDCNY"
)


with open("ParamEst_ER.tex", "w") as f:
    f.write(DFLatex)

print("done")
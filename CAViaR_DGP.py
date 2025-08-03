import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import t
from arch.univariate.distribution import SkewStudent
from scipy.integrate import quad
from scipy.special import beta

"""  PARAMETRI DI BASE """
BurnIn = 10**4
PostBurnIn = 200
N = BurnIn + PostBurnIn
K = 1000 # lag massimo --> da inf viene troncato
Presample = np.zeros(K) # valori fuori dal campione vengono inizializzati a 0
ALPHA = 0.01

# definizioni indicatrici negativa e positiva
def I_m(x):
    return 1 if x < 0 else 0

def I_p(x):
    return 1 if x > 0 else 0

"""  SPECIFICAZIONI CAViaR -> ritornano il VaR al tempo t secondo i vari modelli """

# t: tempo a cui ci si trova
# v_z: valore del VaR
# Z_t: vettore con tutti i valori del pre-campioni (imposti a 0) e del campione generato per il BurbIn e per il PostBurnIn
# HT: Heavy Tails, se una distribuzione ha code pesanti, come la t-Stud(df=5) allora è meglio imporre un controllo sulla TotalSum


def CSASAV(t, v_z, Z_t):
    b0, b1, b2, b3 = 0.05*abs(v_z), 0.8, 0.15*abs(v_z), 0.9

    q = 1 / b3 - 1
    p = q * b1 / (1 - b1)

    ks = np.arange(1, K)
    Z_ts = np.abs(Z_t[t - ks])  

    cost = ((1 - b1) * beta(p, q))
    arg = beta(p - 1 + ks, q + 1) / cost
    
    TotalSum = np.dot(arg, Z_ts)

    v_t = - (b0 / (1 - b1) + b2 * TotalSum)

    return v_t


def CSAAS(t, v_z, Z_t):
    b0, b1, b2m, b2p, b3 = 0.05*abs(v_z), 0.8, 0.25*abs(v_z), 0.05*abs(v_z), 0.9

    q = 1 / b3 - 1
    p = q * b1 / (1 - b1)

    ks = np.arange(1, K)
    Z_ts = np.abs(Z_t[t - ks])

    I_m = (Z_t[t - ks] < 0).astype(float)
    I_p = (Z_t[t - ks] > 0).astype(float)

    cost = ((1 - b1) * beta(p, q))
    arg = beta(p - 1 + ks, q + 1) / cost

    Z_ts_I = (b2m * I_m + b2p * I_p) * Z_ts

    TotalSum = np.dot(arg, Z_ts_I)

    v_t = - ((b0 / (1 - b1)) + TotalSum)

    return v_t


def CSAIG(t, v_z, Z_t):
    b0, b1, b2, b3 = 0.05*v_z**2, 0.8, 0.15*v_z**2, 0.9

    q = 1 / b3 - 1
    p = q * b1 / (1 - b1)

    ks = np.arange(1, K)
    Z_ts = np.array(Z_t[t - ks]**2)  

    cost = ((1 - b1) * beta(p, q))
    arg = beta(p - 1 + ks, q + 1) / cost
    
    TotalSum = np.dot(arg, Z_ts)

    v_t = - np.sqrt(b0 / (1 - b1) + b2 * TotalSum)

    return v_t


def SAV(t, v_z, Z_t):
    b0, b1, b2 = 0.05*abs(v_z), 0.8, 0.15*abs(v_z)

    ks = np.arange(1, K)
    Z_ts = np.abs(Z_t[t - ks])  
    b1s = b1**(ks-1)    
    
    TotalSum = np.dot(b1s, Z_ts)

    v_t = - (b0 / (1 - b1) + b2 * TotalSum)
    
    return v_t
    

def AS(t, v_z, Z_t):
    b0, b1, b2m, b2p = 0.05*abs(v_z), 0.8, 0.25*abs(v_z), 0.05*abs(v_z)

    ks = np.arange(1, K)
    Z_ts = np.abs(Z_t[t - ks])
    b1s = b1**(ks-1) 

    I_m = (Z_t[t - ks] < 0).astype(float)
    I_p = (Z_t[t - ks] > 0).astype(float)

    Z_ts_I = (b2m * I_m + b2p * I_p) * Z_ts

    TotalSum = np.dot(b1s, Z_ts_I)

    v_t = - ((b0 / (1 - b1)) + TotalSum)

    return v_t


def IG(t, v_z, Z_t):
    b0, b1, b2 = 0.05*v_z**2, 0.8, 0.15*v_z**2

    TotalSum = 0
    for k in range(1, K, 1):
        TotalSum += b1**(k-1) * Z_t[t-k]**2
    
    v_t = - np.sqrt(b0/(1-b1) + b2*TotalSum) 

    return v_t


def ChooseSpecific(specif, z_t, v_z):
    # concateno una sola volta
    Z_t = np.concatenate((Presample, z_t))
    V_t = np.zeros(N)

    mapping = {
        "SAV": SAV,
        "AS":  AS,
        "IG":  IG,
        "CSA-SAV": CSASAV,
        "CSA-AS":  CSAAS,
        "CSA-IG":  CSAIG,
    }

    try:
        func = mapping[specif]
    except KeyError:
        raise NotImplementedError(f"Specificazione {specif} non implementata")

    for t in range(len(z_t)):
        V_t[t] = func(t=t+K, v_z=v_z, Z_t=Z_t)

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
    V_t = ChooseSpecific(specif=specif, z_t=z_t, v_z=v_z)

    # 5 - Calcolare e_t, a partire da v_t
    E_t = (1 + gamma) * V_t

    # 6 - Calcolare y_t
    Y_t = (V_t / v_z) * z_t

    return Y_t, V_t, E_t



"""  Visualizzazione grafica """
# impostazioni stile
plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in"
    }
)


def plot_model(ax, i, pdf, specif, alpha=ALPHA, seed=0, ylim=(-9, 5.5)):
    # genera i dati
    Y, V, E = CSA_CAViaR_FZ(pdf=pdf, alpha=alpha, specif=specif, seed=seed)

    # titolo automatico
    dist_name = "Normale" if pdf == "Norm" else "t-Student"
    title = f"{specif}, {dist_name}"

    # plot
    x = np.arange(0, PostBurnIn)
    ax.plot(x, Y[-PostBurnIn:], linewidth=1, color='#5356FF', label="Y_t")
    ax.plot(x, V[-PostBurnIn:], linewidth=1, color='#FFCB61', label="V_t")
    ax.plot(x, E[-PostBurnIn:], linewidth=1, color='#ED3500', label="E_t")
    ax.set_ylim(*ylim)
    ax.set_title(f"({i+1}) {title}", fontsize=8, fontweight='bold')
    ax.tick_params(labelsize=6)
    ax.grid(True)
    ax.ticklabel_format(style='plain', axis='y')
    ax.legend(loc="upper right", fontsize=4)


plot_configs = [
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


fig, axs = plt.subplots(4, 3, figsize=(8, 10)) 
axs = axs.ravel()

for i, (pdf, specif, seed) in enumerate(plot_configs):
    plot_model(ax=axs[i], i=i, pdf=pdf, specif=specif, seed=seed)

plt.tight_layout()
plt.savefig("CAViaR_DGP01.png", dpi=300, bbox_inches='tight')
plt.show()

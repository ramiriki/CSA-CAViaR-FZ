import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox
from hurst import compute_Hc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from scipy.special import gamma


# Calcola (1 - L)^d |y_t| con d = 0.5 per default.
def fractional_diff(y, d, thresh=1e-5, max_lags=30):
    # Step 1: calcola coefficienti
    coeffs = [1.0]
    for k in range(1, max_lags):
        coeff = (-1)**k * gamma(d + 1) / (gamma(k + 1) * gamma(d - k + 1))
        if abs(coeff) < thresh:
            break
        coeffs.append(coeff)
    coeffs = np.array(coeffs)

    # Step 2: applica convoluzione
    y_abs = np.abs(y)
    result = np.convolve(y_abs, coeffs[::-1], mode='valid')
    return result



# Calcola statistica di Ljung-Box
def LjungBox(y, lag):
    return acorr_ljungbox(y, lags=[lag], return_df=True)["lb_stat"].values[0] # lb_pvalue per p_value



# Calcola esponente di Hurst
def HurstExp(y, kind):
    hurst_exp, _, _ = compute_Hc(y, kind=kind)
    return hurst_exp



# Calcola Corr(y1_t, y2_{t-1})
def CrossAutocorrLag1(y1, y2):
    y1 = np.asarray(y1).astype(float)
    y2 = np.asarray(y2).astype(float)

    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    # Allinea: y_t con x_{t-1}
    y1_t = y1[1:]
    y2_tm1 = y2[:-1]

    # Rimuove eventuali NaN
    mask = ~np.isnan(y1_t) & ~np.isnan(y2_tm1)
    y1_t = y1_t[mask]
    y2_tm1 = y2_tm1[mask]

    return np.corrcoef(y1_t, y2_tm1)[0, 1]



# caricamento dati
df = pd.read_csv(
    "StockIndices.csv",
    index_col=0,
    parse_dates=True
)

Indices = ["Returns_FTSE 100", "Returns_HSI", "Returns_Nikkei 225", "Returns_S&P 500"]

# statistiche di una valuta
def StatsForCurrency(indice):
    y_t = df[indice].dropna()
    y_t_abs = np.abs(y_t)
    y_t_diff1 = np.diff(y_t_abs)
    y_t_diff05 = fractional_diff(y=y_t_abs, d=0.5)

    curr_val = [
        np.mean(y_t),
        np.std(y_t),
        skew(y_t),
        kurtosis(y_t, fisher=True),
        adfuller(y_t)[0],
        adfuller(y_t_abs)[0],
        adfuller(y_t_diff1)[0],
        adfuller(y_t_diff05[~np.isnan(y_t_diff05)])[0],
        LjungBox(y=y_t, lag=30),
        LjungBox(y=y_t_abs, lag=30),
        LjungBox(y=y_t_diff1, lag=30),
        LjungBox(y=y_t_diff05[~np.isnan(y_t_diff05)], lag=30),
        HurstExp(y=y_t, kind="change"),
        HurstExp(y=y_t_abs, kind="change"),
        HurstExp(y=y_t_diff1, kind="random_walk"),
        HurstExp(y=y_t_diff05[~np.isnan(y_t_diff05)], kind="random_walk"),
        CrossAutocorrLag1(y1=y_t_abs, y2=y_t_abs),
        CrossAutocorrLag1(y1=y_t_abs, y2=y_t_abs * (y_t < 0)), 
        CrossAutocorrLag1(y1=y_t_abs, y2=y_t_abs * (y_t > 0)), 
        CrossAutocorrLag1(y1=y_t_abs, y2=y_t)
    ]

    return curr_val

# Costruzione tabella con tutte le statistiche
StatsMatrix = [StatsForCurrency(ind) for ind in Indices]

indice = [ind.split("Returns_")[1] for ind in Indices]

# Converti in DataFrame
AllStats = pd.DataFrame(
    np.array(StatsMatrix).T,
    index=["Media", "Std dev", "Skewness", "Kurtosis", 
           "ADF1", "ADF2", "ADF3", "ADF4", 
           "LB1", "LB2", "LB3", "LB4",
           "HE1", "HE2", "HE3", "HE4",
           "CA1", "CA2", "CA3", "CA4"],
    columns=indice
)


latex_index = {
    "Media": r"Mean",
    "Std dev": r"Standard deviation",
    "Skewness": r"Skewness",
    "Kurtosis": r"Kurtosis",
    "ADF1": r"ADF: $y_t$",
    "ADF2": r"ADF: $\lvert y_t \rvert$",
    "ADF3": r"ADF: $(1 - L) \ \lvert y_t \rvert$",
    "ADF4": r"ADF: $(1 - L)^{0.5} \ \lvert y_t \rvert$",
    "LB1": r"LB: $y_t$",
    "LB2": r"LB: $\lvert y_t \rvert$",
    "LB3": r"LB: $(1 - L) \ \lvert y_t \rvert$",
    "LB4": r"LB: $(1 - L)^{0.5} \ \lvert y_t \rvert$",
    "HE1": r"Hurst: $y_t$",
    "HE2": r"Hurst: $\lvert y_t \rvert$",
    "HE3": r"Hurst: $(1 - L) \ \lvert y_t \rvert$",
    "HE4": r"Hurst: $(1 - L)^{0.5} \ \lvert y_t \rvert$",
    "CA1": r"$Corr(\lvert y_t \rvert, \lvert y_{t-1} \rvert)$",
    "CA2": r"$Corr(\lvert y_t \rvert, I(y_{t-1} < 0) \ \lvert y_{t-1} \rvert)$",
    "CA3": r"$Corr(\lvert y_t \rvert, I(y_{t-1} > 0) \ \lvert y_{t-1} \rvert)$",
    "CA4": r"$Corr(\lvert y_t \rvert, y_{t-1})$"
}

AllStats.index = [latex_index[i] for i in AllStats.index]

AllStats.columns = [r"\textbf{FTSE 100}", r"\textbf{HSI}", r"\textbf{Nikkei 225}", r"\textbf{S\&P 500}"]


AllStatsLatex = AllStats.to_latex(
    escape=False,
    column_format="lcccc",
    float_format="%.3f",
    longtable=True,
    caption=r"Statistiche descrittive delle serie dei rendimenti degli indici azionari $y_t$",
    label="tab:SIdescriptivestats"
)


with open("SIDescriptiveStats.tex", "w") as f:
    f.write(AllStatsLatex)

print("done")
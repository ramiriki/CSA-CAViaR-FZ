import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv(
    "StockIndices.csv",
    index_col=0,
    parse_dates=True
)

df.columns = df.columns.str.strip()

Indices = ["Returns_FTSE 100", "Returns_HSI", "Returns_Nikkei 225", "Returns_S&P 500"]

def plot_acorr(ax, indice, ylim=(-0.1, 1), max_lag=50):
    ind = indice.split("Returns_")[1]
    lags = np.arange(max_lag)

    y_t = df[indice].dropna().to_numpy()
    y_t_abs = np.abs(y_t)

    y_t_acorr = sm.tsa.acf(y_t, nlags=len(lags)-1)
    y_t_abs_acorr = sm.tsa.acf(y_t_abs, nlags=len(lags)-1)

    # titolo automatico
    title = f"ACF: {ind}"

    # plot
    x = np.arange(0, max_lag)
    ax.plot(x, y_t_acorr, linewidth=1.5, color='#5356FF', label=r"$y_t$") # forse np.abs(y_t_acorr)
    ax.plot(x, y_t_abs_acorr, linewidth=1.5, color='#ED3500', label=r"$ | y_t |$")
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=6)
    # ax.set_yscale("log")
    ax.grid(True)
    # ax.ticklabel_format(style='plain', axis='y')
    ax.legend(loc="upper right", fontsize=9)


fig, axs = plt.subplots(2, 2, figsize=(8, 10)) 
axs = axs.ravel()

for i, indice in enumerate(Indices):
    plot_acorr(ax=axs[i], indice=indice, ylim=(-0.15, 1), max_lag=50)

# plt.tight_layout()
plt.savefig("SIAcorr.png", dpi=300, bbox_inches='tight')
plt.show()

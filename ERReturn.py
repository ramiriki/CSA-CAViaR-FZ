import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox
from hurst import compute_Hc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv(
    "ExchangeRates.csv",
    index_col=0,
    parse_dates=True
)

df.columns = df.columns.str.strip()  # elimina eventuali spazi nei nomi

Currencies = ["USD/CNY", "EUR/CNY", "100JPY/CNY", "HKD/CNY"]

fig, axs = plt.subplots(4, 2, figsize=(10, 12))
axs = axs.ravel()

# Mostro solo l'anno al momento d'inizio
tick_dates = df.index.to_series().groupby(df.index.year).first()
tick_labels = [d.strftime('%Y') for d in tick_dates]

for i, currency in enumerate(Currencies):
    # Tassi di cambio
    axs[2*i].plot(df.index, df[currency], color="#5356FF")
    axs[2*i].set_xticks(tick_labels)
    axs[2*i].set_title(f"Tassi di cambio {currency}", fontsize=8, fontweight="bold")
    axs[2*i].grid(True)

    # Rendimenti logaritmici * 100
    curr = np.array(df[currency])
    y_t = 100 * np.diff(np.log(curr))
    axs[2*i + 1].plot(df.index[1:], y_t, color="#5356FF")
    axs[2*i + 1].set_xticks(tick_labels)
    axs[2*i + 1].set_title(f"Rendimenti {currency}", fontsize=8, fontweight="bold")
    axs[2*i + 1].grid(True)

    for ax in [axs[2*i], axs[2*i + 1]]:
        # Solo gli anni come etichetta
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # Un tick per anno
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', labelrotation=0, labelsize=6)
        ax.tick_params(axis='y', labelrotation=0, labelsize=6)

plt.tight_layout()
plt.savefig("ERReturn.png", dpi=300, bbox_inches='tight')
plt.show()

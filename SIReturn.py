import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from statsmodels.stats.diagnostic import acorr_ljungbox
from hurst import compute_Hc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv(
    "StockIndices.csv",
    index_col=0,
    parse_dates=True
)

df.columns = df.columns.str.strip()  # elimina eventuali spazi nei nomi

print(df)

Indices = ["Returns_FTSE 100", "Returns_HSI", "Returns_Nikkei 225", "Returns_S&P 500"]

fig, axs = plt.subplots(2, 2, figsize=(10, 12))
axs = axs.ravel()

# Mostro solo l'anno al momento d'inizio
tick_dates = df.index.to_series().groupby(df.index.year).first()
tick_labels = [d.strftime('%Y') for d in tick_dates]

for i, indice in enumerate(Indices):
    # Tassi di cambio
    ind = indice.split("Returns_")[1]
    axs[i].plot(df.index, df[indice], color="#5356FF")
    axs[i].set_xticks(tick_labels)
    axs[i].set_ylim((-0.15, 0.10))
    axs[i].set_title(f"Tassi di cambio {ind}", fontsize=8, fontweight="bold")
    axs[i].grid(True)

    # Solo gli anni come etichetta
    axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # Tutti i numeri nell'asse y con 2 cifre
    axs[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    # Un tick per anno
    axs[i].xaxis.set_major_locator(mdates.YearLocator())
    axs[i].tick_params(axis='x', labelrotation=0, labelsize=6)
    axs[i].tick_params(axis='y', labelrotation=0, labelsize=6)


plt.tight_layout()
plt.savefig("SIReturn.png", dpi=300, bbox_inches='tight')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

df = pd.read_csv(
    "StockIndices.csv",
    index_col=0,
    parse_dates=True
)

df.columns = df.columns.str.strip()

Indices = ["Returns_FTSE 100", "Returns_HSI", "Returns_Nikkei 225", "Returns_S&P 500"]

fig, axs = plt.subplots(2, 2, figsize=(10, 12))
axs = axs.ravel()

for i, indice in enumerate(Indices):
    ind = indice.split("Returns_")[1]
    y_t = df[indice].dropna().to_numpy()
    
    mu, sigma = np.mean(y_t), np.std(y_t)

    # istogramma
    count, bins, _ = axs[i].hist(y_t, bins=70, density=True, alpha=0.6, color='#5356FF', edgecolor='black')

    # normale teorica
    x = np.linspace(min(y_t), max(y_t), 1000)
    pdf = norm.pdf(x, mu, sigma)
    axs[i].plot(x, pdf, linewidth=1.5, linestyle="--", color="#ED3500", label=f'N({mu:.2f}, {sigma:.2f}²)')
    axs[i].set_title(f"Istogramma indice azionario: {ind}", fontsize=8, fontweight="bold")
    axs[i].xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    axs[i].grid(True)

   
plt.tight_layout()
plt.savefig("SIHist.png", dpi=300, bbox_inches='tight')
plt.show()

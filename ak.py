import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from scipy.special import gamma
import matplotlib.ticker as ticker

fig, axs = plt.subplots(6, 3, figsize=(10, 12))
axs = axs.ravel()

params = [
    (0.1, 0.1), (0.1, 1), (0.1, 10),
    (1, 0.1), (1, 1), (1, 10),
    (10, 0.1), (10, 1), (10, 10)
]

x = np.linspace(0, 1, 500)

# Plot delle PDF Beta
for i, (p, q) in enumerate(params):
    y = beta.pdf(x, p, q)
    axs[i].plot(x, y, color='#5356FF')
    axs[i].set_title(f"({i+1}) Beta pdf: p={p}, q={q}", fontsize=8, fontweight='bold')
    axs[i].tick_params(labelsize=6)
    axs[i].grid(True)
    
    # Disattiva notazione scientifica per asse y
    axs[i].ticklabel_format(style='plain', axis='y')
    
    # Nel caso p=1, q=1, f(x)=1 ovunque → forza ylim per evitare etichetta strana
    if p == 1 and q == 1:
        axs[i].set_ylim(0.9, 1.1)

# Funzione a_k(k)
def ak(k, p, q):
    numerator = q * gamma(p + q) * gamma(p - 1 + k)
    denominator = gamma(p) * gamma(p + q + k)
    return np.log(numerator / denominator)

k = np.linspace(1, 50, 500)

# Calcola tutte le serie a_k in anticipo
ak_values = []
for p, q in params:
    ak_values.append(ak(k, p, q))

# Trova il massimo valore globale tra tutte le serie
global_min = min(np.min(serie) for serie in ak_values)

# Plot dei coefficienti a_k con la stessa scala y
for i, ((p, q), y) in enumerate(zip(params, ak_values)):
    axs[i+9].plot(k, y, color='#5356FF')
    axs[i+9].set_title(f"({i+1}) " + r"$\mathbf{ln(a_k)}$: p=" + f"{p}"+r", q=" + f"{q}", fontsize=7, fontweight='bold')
    axs[i+9].tick_params(labelsize=6)
    axs[i+9].grid(True)
    axs[i+9].ticklabel_format(style='plain', axis='y')
    
    # Stessa scala y per tutti
    axs[i+9].set_ylim(global_min, 0)

plt.tight_layout()
plt.savefig("Beta_ln_ak.png", dpi=300, bbox_inches='tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import beta

fig, axs = plt.subplots(2, 2, figsize=(8, 10))
axs = axs.ravel()

b1 = 0.8
b3s = [0.01, 0.3, 0.5, 0.9]
for i, b3 in enumerate(b3s):
    q = 1/b3 - 1
    p = q * b1/(1-b1)
    K = 100

    w_sav    = b1**(np.arange(0, K)-1)     # w_k per SAV
    w_csasav = beta(p-1+np.arange(1, K+1), q+1) / ((1-b1)*beta(p,q))

    title = f"Pesatura: b3={b3}"

    axs[i].plot(w_sav, linewidth=1, color='#5356FF', label="CAViaR")
    axs[i].plot(w_csasav, linewidth=1, color='#ED3500', label="CSA-CAViaR")
    axs[i].set_title(f"({i+1}) {title}", fontsize=8, fontweight='bold')
    axs[i].tick_params(labelsize=6)
    axs[i].grid(True)
    axs[i].ticklabel_format(style='plain', axis='y')
    axs[i].legend(loc="upper right", fontsize=6)
    axs[i].set_yscale("log")

plt.tight_layout()
plt.savefig("ShortLongDecay.png", dpi=300, bbox_inches='tight')
plt.show()
"""
Gaussian Mixture Model 

"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. GENERATE SYNTHETIC DATA ─────────────────────────────────────────────────
np.random.seed(42)

TRUE_K = 3
true_mu    = np.array([-4.0, 1.0, 6.0])
true_sigma = np.array([ 0.8, 1.2, 0.6])
true_theta = np.array([ 0.3, 0.5, 0.2])

N = 300
z = np.random.choice(TRUE_K, size=N, p=true_theta)
y = np.array([np.random.normal(true_mu[z[i]], true_sigma[z[i]]) for i in range(N)])

print(f"Generated {N} observations from a {TRUE_K}-component GMM")
print(f"True means:   {true_mu}")
print(f"True sigmas:  {true_sigma}")
print(f"True weights: {true_theta}")
print()

# ── 2. BUILD AND FIT THE MODEL WITH PyMC ───────────────────────────────────────
import pymc as pm

K = 3

print("Fitting Gaussian Mixture Model...")
print("(This uses NUTS — the same sampler Stan uses)\n")

with pm.Model() as gmm:

    theta = pm.Dirichlet("theta", a=np.ones(K))

    import pytensor.tensor as pt
    mu_raw = pm.Normal("mu_raw", mu=0, sigma=10, shape=K)
    mu = pm.Deterministic("mu", pt.sort(mu_raw))

    sigma = pm.Exponential("sigma", lam=1, shape=K)

    obs = pm.NormalMixture("obs", w=theta, mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.9,
        progressbar=True,
        return_inferencedata=True,
    )

print("\nSampling complete!")

# ── 3. SUMMARIZE POSTERIOR ─────────────────────────────────────────────────────
import arviz as az

summary = az.summary(trace, var_names=["theta", "mu_raw", "sigma"], round_to=3)
print("\nPosterior Summary:")
print(summary)

post_mu    = trace.posterior["mu"].values.reshape(-1, K)
post_sigma = trace.posterior["sigma"].values.reshape(-1, K)
post_theta = trace.posterior["theta"].values.reshape(-1, K)

est_mu    = post_mu.mean(axis=0)
est_sigma = post_sigma.mean(axis=0)
est_theta = post_theta.mean(axis=0)

print(f"\nEstimated means:   {np.round(est_mu, 2)}  (true: {true_mu})")
print(f"Estimated sigmas:  {np.round(est_sigma, 2)}  (true: {true_sigma})")
print(f"Estimated weights: {np.round(est_theta, 2)}  (true: {true_theta})")

# ── 4. VISUALIZE ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor="#0f1117")
fig.suptitle("Gaussian Mixture Model — Bayesian Inference with NUTS",
             color="white", fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = ["#FF6B6B", "#4ECDC4", "#FFE66D"]
BG     = "#1a1d2e"
GRID   = "#2a2d3e"

def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.spines[:].set_color(GRID)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.set_title(title, color="white", fontsize=9, pad=6)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

ax1 = fig.add_subplot(gs[0, :2])
style_ax(ax1, "Data Histogram + Fitted Mixture Density")
x_range = np.linspace(y.min() - 1, y.max() + 1, 500)
ax1.hist(y, bins=40, density=True, color="#ffffff", alpha=0.15, label="Data")
true_density = sum(true_theta[k] * stats.norm.pdf(x_range, true_mu[k], true_sigma[k]) for k in range(TRUE_K))
ax1.plot(x_range, true_density, "w--", linewidth=1.5, alpha=0.6, label="True density")
fitted_density = sum(est_theta[k] * stats.norm.pdf(x_range, est_mu[k], est_sigma[k]) for k in range(K))
ax1.plot(x_range, fitted_density, color="#a855f7", linewidth=2, label="Fitted mixture")
for k in range(K):
    comp = est_theta[k] * stats.norm.pdf(x_range, est_mu[k], est_sigma[k])
    ax1.fill_between(x_range, comp, alpha=0.25, color=COLORS[k])
    ax1.plot(x_range, comp, color=COLORS[k], linewidth=1.2, label=f"Component {k+1}")
ax1.legend(fontsize=7, loc="upper right", facecolor=BG, edgecolor=GRID, labelcolor="white")
ax1.set_xlabel("y", color="#aaaaaa", fontsize=8)
ax1.set_ylabel("density", color="#aaaaaa", fontsize=8)

ax2 = fig.add_subplot(gs[0, 2])
style_ax(ax2, "Posterior: Component Means (μ)")
for k in range(K):
    ax2.hist(post_mu[:, k], bins=40, density=True, alpha=0.6, color=COLORS[k], label=f"μ_{k+1}")
    ax2.axvline(true_mu[k], color=COLORS[k], linestyle="--", linewidth=1.5, alpha=0.9)
ax2.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor="white")
ax2.set_xlabel("μ", color="#aaaaaa", fontsize=8)

ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Posterior: Component Std Devs (σ)")
for k in range(K):
    ax3.hist(post_sigma[:, k], bins=40, density=True, alpha=0.6, color=COLORS[k], label=f"σ_{k+1}")
    ax3.axvline(true_sigma[k], color=COLORS[k], linestyle="--", linewidth=1.5, alpha=0.9)
ax3.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor="white")
ax3.set_xlabel("σ", color="#aaaaaa", fontsize=8)

ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Posterior: Mixing Weights (θ)")
for k in range(K):
    ax4.hist(post_theta[:, k], bins=40, density=True, alpha=0.6, color=COLORS[k], label=f"θ_{k+1}")
    ax4.axvline(true_theta[k], color=COLORS[k], linestyle="--", linewidth=1.5, alpha=0.9)
ax4.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor="white")
ax4.set_xlabel("θ", color="#aaaaaa", fontsize=8)
ax4.text(0.05, 0.95, "Dashed = true value", transform=ax4.transAxes, color="#888888", fontsize=7, va="top")

ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, "MCMC Trace: μ (both chains)")
mu_chains = trace.posterior["mu_raw"].values
for k in range(K):
    for chain in range(mu_chains.shape[0]):
        ax5.plot(mu_chains[chain, :, k], color=COLORS[k], alpha=0.5, linewidth=0.5)
    ax5.axhline(true_mu[k], color=COLORS[k], linestyle="--", linewidth=0.8, alpha=0.7)
ax5.set_xlabel("draw", color="#aaaaaa", fontsize=8)
ax5.set_ylabel("μ value", color="#aaaaaa", fontsize=8)

ax6 = fig.add_subplot(gs[2, 0])
style_ax(ax6, "Convergence: R̂ per parameter\n(good if R̂ < 1.01)")
rhat_data = az.rhat(trace)
params, rhats = [], []
for var in ["theta", "mu_raw", "sigma"]:
    for i, v in enumerate(rhat_data[var].values.flatten()):
        params.append(f"{var}[{i}]")
        rhats.append(float(v))
colors_bar = ["#4ade80" if r < 1.01 else "#f87171" for r in rhats]
ax6.barh(params, rhats, color=colors_bar, alpha=0.8)
ax6.axvline(1.01, color="white", linestyle="--", linewidth=1, alpha=0.5)
ax6.set_xlabel("R̂", color="#aaaaaa", fontsize=8)
for i, (p, r) in enumerate(zip(params, rhats)):
    ax6.text(r + 0.001, i, f"{r:.4f}", va="center", color="white", fontsize=7)

ax7 = fig.add_subplot(gs[2, 1:])
style_ax(ax7, "Posterior Predictive Check\n(50 draws from posterior vs. observed data)")
rng = np.random.default_rng(0)
idx = rng.integers(0, len(post_mu), size=50)
bins = np.linspace(y.min() - 1, y.max() + 1, 50)
for i in idx:
    z_ppc = rng.choice(K, size=N, p=post_theta[i])
    y_ppc = rng.normal(post_mu[i][z_ppc], post_sigma[i][z_ppc])
    ax7.hist(y_ppc, bins=bins, density=True, alpha=0.03, color="#a855f7")
ax7.hist(y, bins=bins, density=True, color="white", alpha=0.5, histtype="step", linewidth=1.5, label="Observed data")
ax7.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor="white")
ax7.set_xlabel("y", color="#aaaaaa", fontsize=8)
ax7.set_ylabel("density", color="#aaaaaa", fontsize=8)

out_path = os.path.join(OUTPUT_DIR, "gmm_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
print(f"\nPlot saved to {out_path}")
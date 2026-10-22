"""
day01_gradient_descent.py
=========================
Linear Regression trained from scratch via Batch Gradient Descent.

Visualises:
  Panel 1 – Data scatter + evolving regression lines (optimal LR)
  Panel 2 – MSE Loss vs Epoch (3 learning rates, log scale)
  Panel 3 – 2D contour of loss surface + optimisation trajectory
  Panel 4 – Parameter convergence (w & b vs true values)

Output: day01_gradient_descent_artifact.png
"""

import sys
import numpy as np

# Force UTF-8 output on Windows (avoids CP1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Global aesthetics
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#f0f6fc",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "font.family":       "DejaVu Sans",
    "font.size":         9,
})

# Accent palette
C_SCATTER   = "#58a6ff"   # data points
C_TRUE_LINE = "#3fb950"   # true regression line
C_SLOW      = "#d29922"   # too-slow LR
C_OPTIMAL   = "#58a6ff"   # optimal LR
C_DIVERGE   = "#f85149"   # diverging LR
C_W         = "#bc8cff"   # weight curve
C_B         = "#ffa657"   # bias curve
C_TRUE_REF  = "#3fb950"   # true-value dashed lines

EPOCH_SNAPSHOTS = [1, 5, 15, 60]          # epochs to show in Panel 1
SNAPSHOT_ALPHAS = [0.30, 0.50, 0.75, 1.0] # opacity per snapshot

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Synthetic data
# ──────────────────────────────────────────────────────────────────────────────
TRUE_W, TRUE_B = 3.5, 2.0
RNG = np.random.default_rng(42)

N = 100
X = RNG.uniform(-2, 2, N)
NOISE = RNG.normal(0, 1.2, N)
Y = TRUE_W * X + TRUE_B + NOISE

# ──────────────────────────────────────────────────────────────────────────────
# 2.  GradientDescentVisualizer
# ──────────────────────────────────────────────────────────────────────────────
class GradientDescentVisualizer:
    """
    From-scratch batch gradient descent for simple linear regression.
    No scikit-learn, no autograd.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 lr: float = 0.08, epochs: int = 60,
                 w_init: float = 0.0, b_init: float = 0.0):
        self.x      = x
        self.y      = y
        self.lr     = lr
        self.epochs = epochs
        self.n      = len(x)

        # learnable parameters
        self.w = w_init
        self.b = b_init

        # history
        self.loss_history = []
        self.w_history    = []
        self.b_history    = []

    # ── forward pass ──────────────────────────────────────────────────────────
    def predict(self, x: np.ndarray) -> np.ndarray:
        """ŷ = w·x + b"""
        return self.w * x + self.b

    # ── loss ──────────────────────────────────────────────────────────────────
    def mse_loss(self, y_hat: np.ndarray) -> float:
        """L = (1/N) Σ (ŷ_i − y_i)²"""
        residuals = y_hat - self.y
        return float(np.mean(residuals ** 2))

    # ── gradients ─────────────────────────────────────────────────────────────
    def _grad_w(self, y_hat: np.ndarray) -> float:
        """∂L/∂w = (2/N) Σ (ŷ_i − y_i)·x_i"""
        return float((2.0 / self.n) * np.dot(y_hat - self.y, self.x))

    def _grad_b(self, y_hat: np.ndarray) -> float:
        """∂L/∂b = (2/N) Σ (ŷ_i − y_i)"""
        return float((2.0 / self.n) * np.sum(y_hat - self.y))

    # ── parameter update ──────────────────────────────────────────────────────
    def _update_params(self, dw: float, db: float) -> None:
        """w ← w − α·∂L/∂w,   b ← b − α·∂L/∂b"""
        self.w -= self.lr * dw
        self.b -= self.lr * db

    # ── training loop ─────────────────────────────────────────────────────────
    def fit(self) -> dict:
        """
        Run batch gradient descent for self.epochs iterations.
        Returns a dict with full history and per-snapshot lines.
        """
        snapshot_lines = {}   # epoch_number → (w_snap, b_snap)

        for epoch in range(1, self.epochs + 1):
            y_hat = self.predict(self.x)
            loss  = self.mse_loss(y_hat)
            dw    = self._grad_w(y_hat)
            db    = self._grad_b(y_hat)
            self._update_params(dw, db)

            # clamp extreme values (for diverging LR display)
            self.w = float(np.clip(self.w, -1e6, 1e6))
            self.b = float(np.clip(self.b, -1e6, 1e6))
            loss   = min(loss, 1e10)

            self.loss_history.append(loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)

            if epoch in EPOCH_SNAPSHOTS:
                snapshot_lines[epoch] = (self.w, self.b)

        return {
            "w_final":        self.w,
            "b_final":        self.b,
            "loss_final":     self.loss_history[-1],
            "loss_history":   self.loss_history,
            "w_history":      self.w_history,
            "b_history":      self.b_history,
            "snapshot_lines": snapshot_lines,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Run experiments
# ──────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = {
    "Too Slow (α=0.002)":       {"lr": 0.002, "color": C_SLOW,    "ls": "--"},
    "Optimal (α=0.08)":         {"lr": 0.08,  "color": C_OPTIMAL, "ls": "-"},
    "Diverging (α=0.48)":       {"lr": 0.48,  "color": C_DIVERGE, "ls": "-."},
}

EPOCHS  = 60
results = {}

for label, cfg in EXPERIMENTS.items():
    gd = GradientDescentVisualizer(X, Y, lr=cfg["lr"], epochs=EPOCHS)
    results[label] = {**gd.fit(), **cfg}

# Shorthand for the optimal experiment
OPT = results["Optimal (α=0.08)"]


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Loss-surface helper
# ──────────────────────────────────────────────────────────────────────────────
def compute_loss_surface(x: np.ndarray, y: np.ndarray,
                         w_range: tuple, b_range: tuple,
                         grid_size: int = 200):
    """Return W_grid, B_grid, Loss_grid for contour plotting."""
    ws = np.linspace(*w_range, grid_size)
    bs = np.linspace(*b_range, grid_size)
    W, B = np.meshgrid(ws, bs)
    n    = len(x)
    Loss = np.array([
        np.mean((w * x + b - y) ** 2)
        for w, b in zip(W.ravel(), B.ravel())
    ]).reshape(W.shape)
    return W, B, Loss


W_RANGE = (-1.0, 5.5)
B_RANGE = (-2.0, 6.0)
W_grid, B_grid, Loss_grid = compute_loss_surface(X, Y, W_RANGE, B_RANGE)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Dashboard
# ──────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13), facecolor="#0d1117")
gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28,
               left=0.06, right=0.97, top=0.93, bottom=0.06)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

fig.suptitle("Linear Regression via Batch Gradient Descent  ·  From Scratch",
             fontsize=15, fontweight="bold", color="#f0f6fc", y=0.97)

# ── shared styling helper ─────────────────────────────────────────────────────
def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, fontweight="bold", color="#f0f6fc", pad=8)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 – Data + evolving regression lines (optimal LR)
# ─────────────────────────────────────────────────────────────────────────────
style_ax(ax1, "Panel 1 – Data & Regression Lines (Optimal α)",
         "x", "y")

ax1.scatter(X, Y, color=C_SCATTER, s=18, alpha=0.55, zorder=2, label="Noisy data")

x_line = np.linspace(X.min() - 0.1, X.max() + 0.1, 300)

# True line
ax1.plot(x_line, TRUE_W * x_line + TRUE_B,
         color=C_TRUE_LINE, lw=1.8, ls=":", zorder=5, label="True line")

# Snapshot lines
for idx, epoch in enumerate(EPOCH_SNAPSHOTS):
    w_s, b_s = OPT["snapshot_lines"][epoch]
    alpha_val = SNAPSHOT_ALPHAS[idx]
    ax1.plot(x_line, w_s * x_line + b_s,
             color=C_OPTIMAL, lw=1.4, alpha=alpha_val, zorder=4,
             label=f"Epoch {epoch}")

ax1.legend(fontsize=7.5, loc="upper left",
           framealpha=0.85, ncol=2)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 – MSE Loss vs Epoch (log scale)
# ─────────────────────────────────────────────────────────────────────────────
style_ax(ax2, "Panel 2 – MSE Loss vs Epoch (log scale)", "Epoch", "MSE Loss")

for label, res in results.items():
    loss_arr = np.array(res["loss_history"], dtype=float)
    # clip tiny or negative values before log
    loss_arr = np.clip(loss_arr, 1e-8, None)
    ax2.plot(range(1, EPOCHS + 1), loss_arr,
             color=res["color"], lw=2, ls=res["ls"],
             label=label, alpha=0.9)

ax2.set_yscale("log")
ax2.legend(fontsize=7.5, loc="upper right", framealpha=0.85)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 – 2D contour loss surface + trajectory (optimal LR)
# ─────────────────────────────────────────────────────────────────────────────
style_ax(ax3, "Panel 3 – Loss Surface & Optimisation Trajectory (Optimal α)",
         "w  (slope)", "b  (intercept)")

levels = np.logspace(np.log10(Loss_grid.min() + 1e-3),
                     np.log10(Loss_grid.max()), 35)

cf = ax3.contourf(W_grid, B_grid, Loss_grid,
                  levels=levels, norm=LogNorm(), cmap="magma", alpha=0.85)
ax3.contour(W_grid, B_grid, Loss_grid,
            levels=levels, norm=LogNorm(), colors="white",
            linewidths=0.3, alpha=0.25)

cbar = fig.colorbar(cf, ax=ax3, pad=0.02, fraction=0.046)
cbar.ax.tick_params(labelsize=7, colors="#8b949e")
cbar.set_label("MSE Loss", fontsize=7.5, color="#8b949e")

# trajectory arrows (thin out if many steps)
ws_traj = [0.0] + OPT["w_history"]
bs_traj = [0.0] + OPT["b_history"]
step    = max(1, len(ws_traj) // 25)

for i in range(0, len(ws_traj) - 1, step):
    dw = ws_traj[i + step] - ws_traj[i]
    db = bs_traj[i + step] - bs_traj[i]
    ax3.annotate("",
                 xy=(ws_traj[i + step], bs_traj[i + step]),
                 xytext=(ws_traj[i], bs_traj[i]),
                 arrowprops=dict(arrowstyle="->",
                                 color=C_OPTIMAL,
                                 lw=1.1, alpha=0.85))

ax3.scatter([0.0], [0.0], color="#f0f6fc",  s=55, zorder=6, label="Start (0,0)")
ax3.scatter([TRUE_W], [TRUE_B], color=C_TRUE_LINE, marker="*",
            s=160, zorder=6, label=f"True ({TRUE_W}, {TRUE_B})")
ax3.scatter([OPT["w_final"]], [OPT["b_final"]], color=C_OPTIMAL,
            marker="X", s=90, zorder=6,
            label=f"Final ({OPT['w_final']:.2f}, {OPT['b_final']:.2f})")

ax3.legend(fontsize=7, loc="upper left", framealpha=0.85)
ax3.set_xlim(W_RANGE)
ax3.set_ylim(B_RANGE)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 – Parameter convergence
# ─────────────────────────────────────────────────────────────────────────────
style_ax(ax4, "Panel 4 – Parameter Convergence (Optimal α)",
         "Epoch", "Value")

epochs_arr = np.arange(1, EPOCHS + 1)

ax4.plot(epochs_arr, OPT["w_history"], color=C_W, lw=2.0, label="w (learned)")
ax4.plot(epochs_arr, OPT["b_history"], color=C_B, lw=2.0, label="b (learned)")

ax4.axhline(TRUE_W, color=C_W,       lw=1.2, ls="--", alpha=0.55,
            label=f"True w = {TRUE_W}")
ax4.axhline(TRUE_B, color=C_B,       lw=1.2, ls="--", alpha=0.55,
            label=f"True b = {TRUE_B}")

ax4.legend(fontsize=7.5, loc="center right", framealpha=0.85)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Save figure
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_PATH = "day01_gradient_descent_artifact.png"
fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"\n\u2705  Figure saved \u2192 {OUTPUT_PATH}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Terminal summary table
# ──────────────────────────────────────────────────────────────────────────────
SEP   = "\u2500" * 68
HEAD  = f"{'Learning Rate':<28} {'Final w':>10} {'Final b':>10} {'Final MSE':>12}"
print(SEP)
print("  GRADIENT DESCENT  \u00b7  Summary after 60 epochs")
print(SEP)
print(f"  {HEAD}")
print(SEP)

for label, res in results.items():
    w_f    = res["w_final"]
    b_f    = res["b_final"]
    loss_f = res["loss_final"]
    # guard against inf/nan for display
    w_str    = f"{w_f:>10.4f}"   if abs(w_f)    < 1e5 else f"{'DIVERGED':>10}"
    b_str    = f"{b_f:>10.4f}"   if abs(b_f)    < 1e5 else f"{'DIVERGED':>10}"
    loss_str = f"{loss_f:>12.4f}" if loss_f      < 1e8 else f"{'DIVERGED':>12}"
    print(f"  {label:<28} {w_str} {b_str} {loss_str}")

print(SEP)
print(f"  {'True values':<28} {'3.5000':>10} {'2.0000':>10} {'':>12}")
print(SEP)
print()

plt.show()

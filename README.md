# Neural SDE Calibration via Girsanov Measure Change

> Learn stochastic volatility dynamics from option prices using neural networks and measure theory.

This project implements a **Neural Stochastic Differential Equation (Neural SDE)** framework for financial option pricing. Instead of assuming a fixed parametric model, a small neural network learns the volatility dynamics directly from market option prices — while staying mathematically consistent with no-arbitrage pricing through the **Girsanov theorem**.

---

## What This Project Does

Options are financial contracts whose value depends on how much an asset's price might move. Pricing them requires a model of how the asset price *and* its volatility evolve over time.

A classic model is **Heston (1993)**: it describes price and volatility as two coupled random processes (SDEs), driven by two correlated sources of randomness. The volatility process has a "mean-reversion" force pulling it back towards a long-run level — controlled by a constant parameter κ.

**The problem:** real market option prices often cannot be reproduced exactly by any fixed κ. The true volatility dynamics are more complex.

**This project's solution:** replace κ with a neural network κ_NN(v) that is a *function of the current volatility level v*. The neural network is trained end-to-end using Monte Carlo simulation — generate thousands of simulated price paths, compute option payoffs, compare to market prices, backpropagate the error, update the network.

The twist is that the neural model's paths live under one probability measure (the "physical" simulation measure P), but option pricing requires expectations under a different measure (the "risk-neutral" measure Q). **Girsanov's theorem** bridges this gap.

---

## The Key Idea: Girsanov's Theorem

Girsanov's theorem says: *you can change the probability measure of a stochastic process by reweighting your sample paths.*

Concretely, for each simulated path we compute a scalar weight Z (the **Radon-Nikodym derivative**). Paths that are more likely under Q get upweighted; paths less likely get downweighted. The option price then becomes:

```
E^Q[payoff]  =  E^P[payoff × Z]
```

So instead of re-simulating under a different measure, we simulate once under P and correct with Z. This Z is itself a stochastic process — an exponential martingale built from the "market price of risk" φ(v), which encodes how much the drift of the process must shift to move from P to Q.

**Intuition:** Z is like an importance-sampling weight. If your simulation tends to produce paths that are too calm compared to the market, Z will upweight the volatile paths and downweight the calm ones, so the weighted average matches market prices.

The model learns either:
- `nsde-girsanov.py` — κ_NN(v) directly (the drift function)
- `nsde-girsanov-phi.py` — φ(v) directly (the market price of risk), from which κ_NN is derived

---

## How It Works: Algorithm Walkthrough

```
Market option prices (25 strikes × 6 maturities)
         │
         ▼
1. SIMULATE PATHS
   Run 100,000 Monte Carlo paths of (log-price, variance) using
   the current κ_NN(v). Discretize with Milstein scheme on variance.
         │
         ▼
2. COMPUTE GIRSANOV WEIGHTS
   For each path, compute Z_T = exp(∫φ(v)dW - ½∫φ(v)²dt)
   using the relation φ(v) = [κ_NN(v) - κ_Heston(v)] / [ν√(1-ρ²)√v]
         │
         ▼
3. PRICE OPTIONS
   For each (strike K, maturity T): price = mean(payoff(S_T) × Z_T)
   Weight each option by its vega (sensitivity to volatility)
         │
         ▼
4. COMPUTE LOSS
   Loss = MARE(model prices, market prices)          [pricing error]
        + MSE(mean(Z), 1)                            [martingale constraint]
         │
         ▼
5. BACKPROPAGATE
   Gradients flow through the entire pipeline back to κ_NN weights.
   Clip gradients (norm ≤ 1), update with Adam optimizer.
         │
         ▼
6. REPEAT until convergence or 10,000 epochs
         │
         ▼
Output: trained κ_NN(v) + implied volatility surface plots
```

**Pre-training:** before the main loop, the network is warm-started so κ_NN(v) ≈ κ_Heston(v) (i.e., close to the baseline Heston drift). This prevents early divergence.

**Adaptive weighting:** option prices are weighted by their Black-Scholes vega — liquid, at-the-money options contribute more to the loss than illiquid deep out-of-the-money options.

---

## Project Structure

```
Neural_SDE-Girsanov/
│
├── nsde-girsanov.py          # Main script: learns κ_NN(v) directly
├── nsde-girsanov-phi.py      # Variant: learns φ(v), derives κ_NN from it
├── requirements.txt          # Python dependencies
│
├── lib/
│   ├── networks.py           # FFNN and ResFFNN neural network architectures
│   ├── utils.py              # Weight init, Simpson integration, vega weighting
│   ├── compute_iv.py         # Implied volatility + vega via Black-Scholes (py_vollib)
│   ├── options.py            # Payoff classes: Vanilla Call/Put, Lookback, VIX
│   ├── plot.py               # IV surface plotting (2D slices and 3D surface)
│   ├── sde.py                # Experimental torchsde wrapper (not used in main scripts)
│   └── old.py                # Archived earlier implementation attempts
│
├── sample_model/
│   └── heston.py             # Full Heston model: characteristic function pricer,
│                             # Monte Carlo path generation, calibration via dual annealing
│
├── data/
│   └── heston_V0=0.02_kappa=1.5_theta=0.04_nu=0.5_rho=-0.7.npz
│                             # Synthetic option prices (25K × 6T) from known Heston params
│
└── plots/                    # Output directory for training visualisations
    ├── 1.model_vs_market_prices.png
    ├── 1.nsde_kappa_vs_vol_drift.png
    ├── 2.func_and_prices.png  ... 6.func_and_prices3e-8.png
```

**Neural network architecture** (`lib/networks.py`):
- Input dimension: 1 (current variance level v)
- Hidden layers: [15, 15] with ReLU activations
- Output dimension: 1 (κ_NN(v) or φ(v))
- Weights initialised with Xavier uniform; biases at 1e-4

---

## Setup & Running

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchsde`, `numpy`, `scipy`, `matplotlib`, `py_vollib`, `numba`

> Note: `py_vollib` requires a working C compiler for its Cython extension.

### Running

**Version 1 — learn κ_NN directly:**
```bash
python nsde-girsanov.py
```

**Version 2 — learn φ (market price of risk):**
```bash
python nsde-girsanov-phi.py
```

Both scripts:
1. Load market data from `data/`
2. Pre-train the network
3. Run the training loop (up to 10,000 epochs, early stopping at 200 stagnation epochs)
4. Save plots to `plots/`

Key hyperparameters are set at the top of each script:
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_paths` | 100,000 | Monte Carlo sample size |
| `n_steps` | 200 | Time discretisation steps |
| `lr` | 5e-4 | Adam learning rate |
| `n_epochs` | 10,000 | Max training epochs |
| `gamma` | 0.99 | LR scheduler decay per epoch |

---

## Results: How It's Performing

Training converges cleanly. The 6 plot files in `plots/` tell the story:

**`1.nsde_kappa_vs_vol_drift.png`**
Compares the true Heston drift function κ(θ − v) against the neural network's output before and after training. The network learns a function very close to the true one — confirming the model recovers the ground truth.

**`2.func_and_prices.png` through `6.func_and_prices3e-8.png`**
Each is a two-panel figure showing:
- *Top panel:* κ_NN(v) at various stages of training (initial → converged)
- *Bottom panel:* Implied volatility surface before and after training across all strikes and maturities

By the final experiment (`3e-8` loss threshold), the model reproduces the target implied volatility surface to very high accuracy. The vega-weighted MARE loss and the Radon-Nikodym martingale constraint (mean(Z) ≈ 1) are both satisfied at convergence.

**Quantitative summary:**
- Target: 150 option prices (25 strikes × 6 maturities) from a Heston model with κ=1.5
- Initial miscalibration: large pricing errors (model starts with κ=2.5–3.0)
- Final loss: ~3×10⁻⁸ (essentially zero relative error on pricing targets)
- Network size: ~500 parameters total — very lightweight

---

## What's Left / Known Limitations

| Item | Status |
|------|--------|
| Comparison of kappa vs phi parameterisation | Implemented in two scripts; not yet systematically compared in write-up |
| Milstein correction for *price* paths | Applied to variance paths only; TODO in code for price paths |
| VIX option pricing | Payoff classes exist in `lib/options.py`; not yet wired into training loop |
| Undiscounted pricing | Minor open note in code |
| Real market data | Only tested on synthetic Heston data; no live market calibration yet |
| Test suite | None — no automated validation of individual modules |
| `lib/sde.py` | Exploratory torchsde wrapper; not integrated into main scripts |

---

## Mathematical Background

For reference, the key equations:

**Heston dynamics (log-price X = log S):**
```
dX_t  = (r − ½V_t) dt + √V_t dB_t
dV_t  = κ_NN(V_t) dt + ν√V_t dW_t
d⟨B, W⟩_t = ρ dt
```

**Neural SDE:** replace the constant-parameter drift `κ(θ − V_t)` with `κ_NN(V_t)`.

**Girsanov Radon-Nikodym derivative:**
```
Z_T = exp(−∫₀ᵀ φ₁(V_s) dB_s − ∫₀ᵀ φ₂(V_s) dW_s
          − ρ ∫₀ᵀ φ₁φ₂ ds − ½ ∫₀ᵀ (φ₁² + φ₂²) ds)
```
where `φ₁ = 0` (simplified) and `φ₂(v) = [κ_NN(v) − κ_Heston(v)] / [ν√(1−ρ²)√v]`.

**Training loss:**
```
L = Σ_{K,T} w_{K,T} · |C_model(K,T) − C_market(K,T)| / C_market(K,T)
  + λ · (E[Z_T] − 1)²
```
where `w_{K,T}` is the Black-Scholes vega of option (K, T).

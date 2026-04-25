# NeuralSDE-Girsanov

Calibration of stochastic volatility models using neural SDEs and Girsanov's theorem. The Heston model's mean-reversion coefficient $\kappa$ is replaced by a neural network $\kappa^{\mathrm{NN}}(v)$ trained directly on option prices. Rather than re-simulating under the risk-neutral measure, paths are generated once under $\mathbb{P}$ and reweighted via a Girsanov density, keeping the full pipeline differentiable and the measure change explicit.

Two parameterisations are explored: learning $\kappa^{\mathrm{NN}}$ directly (`nsde-girsanov.py`), and learning the market price of risk $\varphi$ from which $\kappa^{\mathrm{NN}}$ is recovered (`nsde-girsanov-phi.py`).

## Model

The neural SDE evolves the log-price $X_t = \log S_t$ and variance $V_t$ as

$$
dX_t = \left(r - \tfrac{1}{2}V_t\right)dt + \sqrt{V_t}\,dB_t, \qquad dV_t = \kappa^{\mathrm{NN}}(V_t)\,dt + \nu\sqrt{V_t}\,dW_t,
$$

with $d\langle B, W\rangle_t = \rho\,dt$. This is a Heston-type model with the drift $\kappa(\theta - v)$ replaced by a small feedforward network evaluated at the current variance level.

The Girsanov density connecting the simulation measure $\mathbb{P}$ to the pricing measure $\mathbb{Q}$ is

$$
Z_T = \exp\!\left(-\int_0^T \varphi_2(V_s)\,dW_s - \frac{1}{2}\int_0^T \varphi_2(V_s)^2\,ds\right),
$$

where the market price of risk in the variance direction is

$$
\varphi_2(v) = \frac{\kappa^{\mathrm{NN}}(v) - \kappa(\theta - v)}{\nu\sqrt{1 - \rho^2}\,\sqrt{v}}.
$$

Option prices are computed as importance-weighted Monte Carlo estimates,

$$
C^{\mathrm{model}}(K, T) = \mathbb{E}^{\mathbb{P}}\!\left[\max(S_T - K,\, 0) \cdot Z_T\right],
$$

and the network is trained by minimising a vega-weighted squared relative pricing error plus a penalty enforcing $\mathbb{E}[Z_T] = 1$:

$$
\mathcal{L} = \sum_{K,T} w_{K,T}\,\frac{\bigl(C^{\mathrm{model}}(K,T) - C^{\mathrm{mkt}}(K,T)\bigr)^2}{C^{\mathrm{mkt}}(K,T)} \;+\; \lambda\,\bigl(\mathbb{E}[Z_T] - 1\bigr)^2.
$$

The variance paths are discretised with a Milstein scheme; $Z_T$ is computed in multiplicative form with the corresponding Milstein correction.

## Structure

```
├── nsde-girsanov.py        # learns κ_NN(v) directly
├── nsde-girsanov-phi.py    # learns φ(v), recovers κ_NN from it
├── requirements.txt
├── lib/
│   ├── networks.py         # FFNN and residual FFNN
│   ├── compute_iv.py       # Black-Scholes IV and vega (py_vollib)
│   ├── options.py          # call/put/lookback/VIX payoffs
│   ├── utils.py            # path utilities, Simpson integration
│   └── plot.py             # IV surface plots
├── sample_model/
│   └── heston.py           # analytic Heston pricer, path generation, calibration
└── data/
    └── heston_V0=0.02_kappa=1.5_theta=0.04_nu=0.5_rho=-0.7.npz
```

The network is a two-hidden-layer feedforward net $\mathbb{R} \to \mathbb{R}$ with 15 units per layer and ReLU activations, taking the current variance $v$ as its sole input. Weights are Xavier-initialised; the network is pre-trained to match the baseline Heston drift before the main training loop.

## Usage

```bash
pip install -r requirements.txt
```

```bash
python nsde-girsanov.py        # kappa parameterisation
python nsde-girsanov-phi.py    # phi parameterisation
```

Both scripts load market prices from `data/`, run up to 10,000 training epochs with early stopping, and write figures to `plots/`.

## Results

Training converges on the synthetic Heston surface (25 strikes $\times$ 6 maturities, $\kappa^{\mathrm{mkt}} = 1.5$, model initialised at $\kappa = 2.5$–$3.0$). The final weighted loss reaches $\approx 3 \times 10^{-8}$, and the learned $\kappa^{\mathrm{NN}}(v)$ closely recovers the true linear drift $\kappa(\theta - v)$. The plots in `plots/` show the evolution of $\kappa^{\mathrm{NN}}$ alongside the implied volatility surface at successive training checkpoints.

## Known limitations

- Tested only on synthetic Heston data; no real market surfaces.
- Feller condition ($2\kappa\theta > \nu^2$) is violated for the default parameters — paths can reach $V = 0$, requiring post-hoc clamping that biases the Girsanov weights.
- $\varphi_1 = 0$ is assumed throughout (no measure change in the price direction).
- Milstein correction for the *price* path is not yet implemented.
- VIX option pricing infrastructure exists in `lib/options.py` but is not wired into the training loop.
- No random seeds are set; results are not reproducible across runs.

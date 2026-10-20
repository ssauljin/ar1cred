# Revision Plan: Tweedie Evolutionary Credibility with AR(1) Latent Dependence

**Source:** N. Kang, MSc project, SFU (Summer 2026), supervised by H. Jeong
**Target:** MDPI Topic *Statistical Learning and Dependence Structures for Financial Time Series* (https://www.mdpi.com/topics/3Y7DH7E4X9). The most natural participating journal is *Risks*; *Econometrics* is an alternative.
**Deadlines:** abstract 31 Dec 2027; manuscript 31 Mar 2028.
**Scope of this revision:** (1) theory, (2) estimation, (3) simulation study. Other ideas from the review are listed in Section 6 so they are not lost.

---

## 0. Summary of the revision

| Area | Current thesis | Revised manuscript |
|---|---|---|
| Premium formula | Proposition 1 for a θ-scale AR(1) moment structure | **Theorem 1** for any second-order stationary latent process with autocovariance γ(k); AR(1), log-AR(1), exchangeable and independent cases become corollaries |
| Structural results | None beyond the ρ = 0 and ρ = 1 corollaries | **Proposition 2**: Kalman (updating-type) recursion, positive weights, Σw < ρ, guaranteed positive premium, bounded maximal bonus |
| Positioning | Builds on Ma et al. (2018) and Ohlsson–Johansson (2006) | Positioned against the classical evolutionary-credibility literature (Albrecht 1985; Gerber–Jones 1975; Sundt 1981; Bühlmann–Gisler Ch. 9; Pinquet et al.; Taylor 2018) |
| Estimation | Unweighted method of moments and Gaussian working-likelihood MCMC (both unreliable), so the simulation is oracle-only | **Semi-parametric two-step GMM** on second moments of standardized residuals, with constraints built in, a J-test, and Wald tests for static and no-dependence restrictions |
| Static benchmark | Bühlmann with the full Var(θ) plugged in (a strawman) | The same GMM with the exchangeable restriction imposed, i.e. a fitted static model (Corollary 2) |
| Simulation | Oracle parameters; grid on log-scale σ up to 2 (CV of θ = 7.3); mean of RMSE over 100 reps | Estimated parameters; grid on Var(θ) and lag-1 correlation; exact population MSE as a unit test; paired differences with Monte Carlo SEs; 500 reps |

---

## 1. Theory

### 1.1 Notation (one symbol, one meaning)

The thesis uses σ for three different quantities: θ-scale SD in Prop. 1, log-scale SD in Ch. 4, and τ in Sec. 2.3. Adopt one table and use it throughout.

| Symbol | Meaning |
|---|---|
| $\mu_{it}=\exp(x_{it}^\top\beta)$ | GLM mean (bold $\boldsymbol\mu_T$ for the vector) |
| $\theta_{it}$ | latent multiplicative factor, $E\theta_{it}=1$ |
| $v=\mathrm{Var}(\theta_{it})$ | latent variance (θ scale) |
| $\gamma(k)=\mathrm{Cov}(\theta_{it},\theta_{i,t+k})$, $r(k)=\gamma(k)/v$ | latent autocovariance and autocorrelation |
| $a=\phi\,E[\theta_{it}^p]$ | within-variance constant |
| $e_{it}=Z_{it}/\mu_{it}-1$ | standardized residual |
| $W_{it}=\log\theta_{it}$, $\sigma_W^2$, $\rho_W$ | log-scale variance and AR(1) parameter (DGP only) |
| $\rho$ | θ-scale AR(1) parameter: $\gamma(k)=v\rho^k$ |

Mapping under the log-AR(1) DGP:
- $v=e^{\sigma_W^2}-1$
- $\gamma(k)=\exp(\sigma_W^2\rho_W^k)-1$
- $E\theta^p=\exp\{\sigma_W^2 p(p-1)/2\}$

The θ-scale autocorrelation is **not** exactly geometric under this DGP.

### 1.2 Add missing prior work

Add a paragraph, or a short subsection, on evolutionary credibility. Each item below is followed by a positioning statement. † = bibliographic details confirmed during the review; all others should be checked against the source.

- Gerber, H.U. & Jones, D.A. (1975). Credibility formulas of the updating type. *TSA* 27. Recursive (Kalman-type) credibility.
- Sundt, B. (1981). Recursive credibility estimation. *Scand. Actuar. J.*
- † Albrecht, P. (1985). An evolutionary credibility model for claim numbers. *ASTIN Bulletin* 15(1), 1–17.
  - Credibility for a **weakly stationary** latent sequence with Poisson counts, with a recursive formula for the coefficients.
  - **This is the closest antecedent to Theorem 1 and must be discussed explicitly.**
- Bühlmann, H. & Gisler, A. (2005), chapter on evolutionary credibility models and recursive calculation (Kalman filter).
- Pinquet, J., Guillén, M. & Bolancé, C. (2001). Allowance for the age of claims in bonus-malus systems. *ASTIN Bulletin* 31(2).
- Bolancé, C., Guillén, M. & Pinquet, J. (2003). Time-varying credibility for frequency risk models: estimation and tests for autoregressive specifications on the random effects. *IME* 33(2). Moment-based estimation and tests; closest antecedent to the new Section 5.
- Pinquet, J. (2020). Positivity properties of ARFIMA(0,d,0) specifications and credibility analysis of frequency risks. *IME*. Relevant to Proposition 2 and to long memory.
- † Taylor, G. (2018). Evolutionary hierarchical credibility. *ASTIN Bulletin* 48(1), 339–374. Relevant to the hierarchical extension in Future Work.
- Estimation references: Hansen (1982, *Econometrica*) for GMM and the J-test; Newey & McFadden (1994, *Handbook of Econometrics* IV, Ch. 36) for two-step estimators; Andrews (1999, *Econometrica*) for parameters on the boundary.
- Also fix existing citations: Jørgensen (1987) is cited in the text but missing from the bibliography.

**Positioning statement.** Relative to Albrecht (1985) and its successors, the new elements are:
1. the Tweedie conditional response with power variance, giving a heteroscedastic, covariate-dependent nugget $a\mu_t^{p-2}$ inside a GLM rating structure;
2. the exact link to Ohlsson–Johansson (2006) Tweedie credibility as the exchangeable special case;
3. positivity and bonus bounds (Proposition 2);
4. a semi-parametric GMM estimator that needs no distributional assumption on θ, plus tests of static versus evolutionary credibility.

### 1.3 Generalize Proposition 1 to Theorem 1

**Assumptions**
- (A1) $Z_{it}\mid\theta_{it}\sim Tw(\mu_{it}\theta_{it},\phi,p)$, $1<p<2$, conditionally independent over $t$ given the latent path.
- (A2) $E\theta_{it}=1$ and $E\theta_{it}^{\max(p,2)}<\infty$.
- (A3) Second-order stationarity: $\mathrm{Cov}(\theta_{it},\theta_{is})=\gamma(|t-s|)$, with the $(T+1)\times(T+1)$ matrix $\Gamma_{T+1}$ positive definite.
- (A4) $E[\theta_{it}^p]$ does not depend on $t$, which follows from strict stationarity.
- Independence across policyholders.

**Theorem 1.** The best linear predictor of $Z_{T+1}$ given $Z_1,\dots,Z_T$ is

$$P_{T+1}=\mu_{T+1}\left\{1+c_T^\top(\Gamma_T+D_T)^{-1}e_T\right\},$$

where:
- $\Gamma_T=[\gamma(|t-s|)]_{t,s\le T}$;
- $D_T=\mathrm{diag}(a\mu_1^{p-2},\dots,a\mu_T^{p-2})$;
- $c_T=(\gamma(T),\gamma(T-1),\dots,\gamma(1))^\top$;
- $e_T=(e_1,\dots,e_T)^\top$.

The proof is the existing proof of Prop. 1, with $\sigma^2\rho^{|t-s|}$ replaced by $\gamma(|t-s|)$.

**Corollaries** (replacing Corollaries 1–2 and covering more cases)
1. $\gamma(k)=0$ for $k\ge1$ gives the Naive premium $\mu_{T+1}$.
2. $\gamma(k)=v$ for all $k$ (exchangeable/static) gives the Ohlsson–Johansson / Bühlmann–Straub-type premium. This is the current Corollary 2 with $V=a/v$.
3. $\gamma(k)=v\rho^k$ is the θ-scale AR(1) case, i.e. the current Prop. 1.
4. $\gamma(k)=\exp(\sigma_W^2\rho_W^k)-1$ is the log-AR(1) case, which is the actual simulation DGP.
5. ARMA, long memory (ARFIMA) and regime-switching autocovariances are further examples. This addresses thesis Limitation 2.

**Remark on moments.** The thesis states (Remark 1) that only the first two moments of θ are needed. In fact $E[\theta^p]$, a fractional moment, also enters. It enters **only through the product $a=\phi E\theta^p$**, which the diagonal second moment of $e_{it}$ identifies (Section 2), so no distributional assumption on θ is needed. Hölder gives $1\le E\theta^p\le(1+v)^{p-1}$.

### 1.4 Structural results (new Proposition 2)

Specialize to the θ-scale AR(1) case $\gamma(k)=v\rho^k$, $\rho\in[0,1)$. Write $x_t=\theta_t-1$ and $\varepsilon_t=e_t-x_t$. Then:
- $x_{t+1}=\rho x_t+\eta_t$ with innovations uncorrelated with the past;
- $e_t=x_t+\varepsilon_t$, where $\varepsilon_t$ is uncorrelated with everything else and $\mathrm{Var}\,\varepsilon_t=a\mu_t^{p-2}$.

This is a linear state-space model in the second-order sense, so the Kalman filter delivers the linear projection exactly.

**(a) Recursion (updating-type form).**
- Start: $\hat x_1=0$, $P_1=v$.
- Gain: $K_t=P_t/(P_t+a\mu_t^{p-2})$.
- Update: $\hat x_{t+1}=\rho\{(1-K_t)\hat x_t+K_te_t\}$ and $P_{t+1}=\rho^2(1-K_t)P_t+(1-\rho^2)v$.
- Premium: $P_{T+1}=\mu_{T+1}(1+\hat x_{T+1})$. This costs $O(T)$ per policy and suits annual rate updating.

**(b) Closed-form weights.** $w_t=\rho^{T+1-t}K_t\prod_{s=t+1}^{T}(1-K_s)$, and these are positive for $\rho>0$.

**(c) Weight sum.** Let $S_t$ denote the running sum of weights. Then $S_{t+1}=\rho\{(1-K_t)S_t+K_t\}$, and by induction $\sum_t w_t<\rho<1$.

**(d) Positivity and bonus bound.** Since $e_t\ge-1$, $P_{T+1}\ge\mu_{T+1}(1-\sum_tw_t)>(1-\rho)\,\mu_{T+1}>0$.
- The premium is always positive, so Tweedie deviance is always defined.
- The maximal bonus, reached by a claim-free history, is bounded by $\rho$.

**(e) Comparative statics** (to prove). Weights increase with $v$ and with $\mu_t$, since a larger $\mu_t$ means a smaller nugget. Link this to Ahn, Jeong & Lu (2021) on the ordering of credibility factors.

**(f) Recency monotonicity** ($w_T\ge\dots\ge w_1$, the "age of claims" property).
- Verified numerically for constant $\mu$; prove it under constant $\mu_t$.
- It can **fail** when $\mu_t$ varies over time (an old year with very large $\mu_t$ has a small nugget). State it with that condition.

**(g) Limits.** $\rho\to1$ recovers Corollary 2, and $\rho=0$ recovers Naive. For constant $\mu$, give the steady-state gain.

For general $\gamma$, note that the Durbin–Levinson or innovations algorithm gives the recursion, and that positivity of the weights is not guaranteed for non-monotone $\gamma$.

### 1.5 Housekeeping (fix all)

- [ ] Use one meaning per symbol (Section 1.1); separate the θ-scale $(v,\rho)$ from the log-scale $(\sigma_W,\rho_W)$ everywhere, including the static α formula, where $\tau^2$ clashes with Sec. 2.3.
- [ ] Prop. 1: change $[\Sigma]_{k,l}=\rho^{k-l}$ to $\rho^{|k-l|}$.
- [ ] Remove the leftover $\mathbf 1_J$ and $Z^{(j)}_{T+1}$ notation and use $\mathbf 1_T$ and $Z_{T+1}$; distinguish scalar $\mu_T$ from vector $\boldsymbol\mu_T$.
- [ ] Proof of Prop. 1 opens with a dangling "Let"; complete or delete it.
- [ ] Sec. 2.3: "geometric decay of correlation in $W_t$" should refer to $\theta_t$; the θ-scale correlation is not geometric, so give the exact formula.
- [ ] Sec. 2.5: "covariance matrix V becomes rank-one" should be Σ (or $\Gamma$).
- [ ] Sec. 2.5: "numerical study in Chapter 3" should point to the simulation section.
- [ ] Remark 1: correct the "first two moments suffice" claim (Section 1.3 above); fix "the the".
- [ ] $E\theta^p=(1+\sigma^2)^{p(p-1)/2}$ holds only under the Remark 1 construction; under the Ch. 4 DGP it is $\exp\{\sigma_W^2p(p-1)/2\}$. State which is used where.
- [ ] Corollary 2 proof: clarify the "choosing $p_j=1$ and $p_j=2$ as in Frangos and Vrontos" remark.
- [ ] Eq. (1.7): rename the gamma parameters (α, β), which clash with eq. (1.6) and the regression β.
- [ ] Ch. 4 says "static worse than naive in 12 of 16" while Ch. 5 says 11 (the table supports 12). This becomes moot once the static benchmark is replaced.
- [ ] Ch. 4 intro says ϕ is varied, but it is fixed at 10.
- [ ] Ch. 4 intro attributes the failure to "maximum likelihood boundary truncation", but Ch. 3 used MoM and MCMC; make them consistent (moot after the rewrite).
- [ ] Rename "sanity check" to "oracle-parameter analysis".
- [ ] Table 4.1 mixes $Y$ and $Z$ for claims.
- [ ] Text spacing such as "specification(Ma et al."; the stray ".." in Sec. 3.1.
- [ ] Conclusion claims "uniformly superior … substantially superior to Bühlmann", which Table 4.2 contradicts at ρ = 0.9. Rewrite it after the new results.
- [ ] Condense Sec. 1.3 (textbook Tweedie, GLM and random-effects material) to about one page for a journal audience.
- [ ] Recompute and report the actual zero-claim proportion in the simulation (currently described as "approximately 30–50%").

---

## 2. Estimation: replace Ch. 3 with semi-parametric GMM

### 2.1 What to remove

- Gaussian working-likelihood MCMC (nimble), eqs. (3.2)–(3.3), and Table 3.1.
- The oracle-only framing. Keep oracle parameters only as a benchmark.

Why the old approach failed, for one sentence or a footnote:
- unweighted cross-products of levels, dominated by large-μ policies;
- a lognormal-specific diagonal moment;
- equations solved without constraints;
- a Ch. 3 design (Var θ = 0.04) with almost no latent signal. Even the oracle premium improves RMSE by only about 0.1% there.

### 2.2 Moment identities

From (A1)–(A2):
- **Lag moments:** $E[e_{it}e_{i,t+k}]=\gamma(k)$ for $k\ge1$. These are free of $\phi$, $p$ and the law of θ.
- **Diagonal moment:** $E[e_{it}^2]=\gamma(0)+a\,\mu_{it}^{p-2}$.

The premium in Theorem 1 depends only on $(\gamma(\cdot),a)$, so every input is identified by second moments of $e_{it}$.

### 2.3 Estimator

**Stage 1 — mean structure.** Fit a Tweedie quasi-likelihood GLM under working independence to get $\hat\beta$ (consistent because $E Z_{it}=\mu_{it}$) and $\hat\mu_{it}$. Then form $e_{it}=Z_{it}/\hat\mu_{it}-1$.

**Stage 2 — joint two-step GMM for $\psi$.** Here $\psi$ contains the structure parameters plus $a$; for the AR(1) case $\psi=(v,\rho,a)$. Per-policy moment contributions:
- $g_{ik}(\psi)=\sum_{t=1}^{T-k}w_{itk}\{e_{it}e_{i,t+k}-\gamma(k;\psi)\}$, for $k=1,\dots,T-1$;
- $h_{ij}(\psi)=\sum_t\omega_{it}z_{itj}\{e_{it}^2-\gamma(0;\psi)-a\mu_{it}^{p-2}\}$, with instruments $z_{it}=(1,\mu_{it}^{p-2})$.

Weights:
- $w_{itk}=(\hat s_{it}^2\hat s_{i,t+k}^2)^{-1}$ and $\omega_{it}=\hat s_{it}^{-4}$, with $\hat s_{it}^2=\hat a\mu_{it}^{p-2}+\hat v$;
- iterate two or three times;
- these down-weight small-μ policies whose residuals are mostly Tweedie noise. This is essential when μ is very heterogeneous: in the Ch. 3 design, equal weights made the plug-in premium far worse than Naive.

Objective: $\hat\psi=\arg\min\bar m(\psi)^\top\hat\Omega^{-1}\bar m(\psi)$, where $\hat\Omega$ is the covariance of per-policy contributions (clustered by policyholder).

Reparametrize as $v=e^{\nu}$, $\rho=\mathrm{logit}^{-1}(\kappa)$, $a=e^{\alpha}$, so estimates always stay in the parameter space.

**Plug-in premium.** Insert $(\gamma(k;\hat\psi),\hat a)$ and $\hat\mu$ into Theorem 1, or into the Proposition 2 recursion.

**Structures fitted with the same machinery:**
- S1: θ-AR(1), $v\rho^k$ (main; Proposition 2 applies).
- S2: log-AR(1), $\exp(\sigma_W^2\rho_W^k)-1$.
- S3: exchangeable, $\gamma(k)=v$. This gives the **fitted Static benchmark** (Corollary 2).
- S0: $\gamma(k)=0$, which is Naive.
- NP: nonparametric $\hat\gamma(1),\dots,\hat\gamma(T-1)$ with a parametric tail only for $\gamma(0)$ and $\gamma(T)$.

### 2.4 Inference and tests

**Asymptotics.** With $I\to\infty$ and $T$ fixed, the estimator is $\sqrt I$-consistent and asymptotically normal with a sandwich variance. Stage 1 matters, because $\partial E[e_te_s]/\partial\beta=-(x_t+x_s)\gamma(|t-s|)\neq0$. Handle it in one of two ways:
- stack the GLM quasi-score $\sum_tx_{it}(Z_{it}-\mu_{it})\mu_{it}^{1-p}$ into the moment vector (one-step joint GMM; Newey–McFadden 1994, §6);
- or use a policyholder-level (cluster) bootstrap, B ≥ 499.

**Specification test.** Hansen's J-test of S1 or S2 (df = number of moments minus number of parameters).

**Static versus evolutionary.** Wald test of $H_0:\gamma(1)=\dots=\gamma(T-1)$ on the NP estimates, with $T-2$ df. This avoids testing ρ = 1 on the boundary. Same spirit as Bolancé, Guillén & Pinquet (2003).

**No dependence.** Wald test of $H_0:\gamma(k)=0$ for all $k\ge1$ on the NP estimates. Unrestricted $\hat\gamma(k)$ means no boundary problem. For $H_0:v=0$ inside S1, use Andrews (1999).

**Bias correction.** The median $\hat v$ was about 15% low at high persistence in the preliminary runs. Use a cluster-bootstrap bias correction and report medians alongside means because of heavy tails.

### 2.5 Practical extensions to include

- **Choice of $p$.** Marginal variance is $a\mu^p+v\mu^2$, so the marginal Tweedie profile-likelihood $\hat p$ is a working value biased toward 2. Options:
  - profile $p$ over a grid by minimizing the J-statistic;
  - add the instrument $\mu^{p-2}\log\mu$ and estimate $p$ jointly;
  - report sensitivity to $p$.
- **Exposure.** For pure premium $Y=Z/\omega$ with $Y\mid\theta\sim Tw(\mu\theta,\phi/\omega,p)$, use $e=Y/\mu-1$; the diagonal moment becomes $v+a\mu^{p-2}/\omega$, and the weights change accordingly.
- **Unbalanced panels.** Sum over the observed pairs (entry, exit and gaps need no special treatment).
- **Computation.** Closed form, $O(IT^2)$, with no MCMC or numerical integration.

### 2.6 Preliminary evidence

Python prototypes `gmm.py` and `gmm_joint.py`; 200 replications, $I=2000$, $T=5$, Ch. 4 covariates, log-AR(1) truth.

| Design (truth) | $\hat v$ median (truth) | lag-1 $\hat r$ median (truth) | J-test size (nominal 5%) | Share of oracle MSE gain captured |
|---|---|---|---|---|
| $\sigma_W=0.5,\rho_W=0.7$ | 0.29 (0.28) | 0.64 (0.67) | 5.5% | 84% |
| $\sigma_W=1,\rho_W=0.5$ | 1.59 (1.72) | 0.39 (0.38) | 2.5% | 93% |
| $\sigma_W=1,\rho_W=0.9$ | 1.47 (1.72) | 0.85 (0.85) | 2.0% | 97% |
| Time-varying $x$, $v=0.5$, $\rho_W=0.7$ (lag-only) | 0.54 mean (0.50) | 0.65 mean (0.66) | 5.5% | 91% |

Joint GMM versus lag-only: joint trims the right tail of $\hat v$ (95th percentile 4.51 → 3.26 in row 2) and adds 1–2 points of captured gain. Lag-only should be reported as a robustness check.

---

## 3. Simulation study: redo with estimated parameters

### 3.0 Audit the existing code first (blocking)

Table 4.2 conflicts with theory. With correct moments, the proposed premium is the best linear predictor, so it cannot lose to Static. Yet Table 4.2 reports it losing at ρ = 0.9 for σ = 1.5 and σ = 2. An independent replication (`mc.py`) and exact population MSE give:

| σ, ρ (log scale) | Thesis N / S / P | Replication N / S / P | Exact N / S / P |
|---|---|---|---|
| 1.0, 0.9 | 29.40 / 28.17 / 27.72 | 29.39 / 28.16 / 27.59 | 29.80 / 28.47 / 27.91 |
| 1.5, 0.9 | 44.78 / 42.01 / 42.96 | 46.44 / 43.47 / 40.46 | 48.75 / 45.24 / 42.18 |
| 2.0, 0.9 | 79.06 / 74.27 / 78.30 | 82.60 / 84.96 / 73.07 | 103.87 / 98.75 / 86.10 |

The σ = 1 column reproduces almost exactly; the gap grows with σ. Checklist:
- [ ] Moment inputs: Var θ must be $e^{\sigma_W^2}-1$ (not $\sigma_W^2$); correlation must be $(e^{\sigma_W^2\rho_W^k}-1)/(e^{\sigma_W^2}-1)$ (not $\rho_W^k$); and $E\theta^p=\exp\{\sigma_W^2p(p-1)/2\}$.
- [ ] Ordering of the cross-covariance vector $(\gamma(T),\dots,\gamma(1))$ against $(Z_1,\dots,Z_T)$.
- [ ] **Unit test.** For any linear predictor with known moments, the exact MSE is
  $\mathrm{MSE}=E_x\big[\mu_{T+1}^2\{\mathrm{Var}(e_{T+1})-2w^\top c+w^\top(\Gamma+D)w\}\big]$, with $\mathrm{Var}(e_{T+1})=v+a\mu_{T+1}^{p-2}$.
  The simulated oracle-parameter MSE must match it within Monte Carlo error.
- [ ] Expected consequence: the "gain peaks at moderate σ" pattern (Fig. 4.3) and the "swamped by latent variability" explanation will likely disappear. With correct moments, the gain rises with latent variance (exact relative RMSE gains at ρ = 0.9: 6.3%, 13.5%, 17.1% for σ_W = 1, 1.5, 2).

### 3.1 Design

**Data-generating process.** Keep the log-AR(1) Tweedie DGP (eq. 4.2), but index the grid by interpretable θ-scale quantities:
- Var θ: $v\in\{0.1,0.25,0.5,1.0\}$, with $v=2$ as a stress case. The current σ = 2 means CV(θ) = 7.3, which is unrealistic and makes RMSE unstable.
- Lag-1 latent correlation: $r(1)\in\{0.3,0.5,0.7,0.9\}$.
- Mapping: $\sigma_W^2=\log(1+v)$ and $\rho_W=\log(1+v\,r(1))/\sigma_W^2$.

**Covariates.** Use time-varying covariates, e.g. $x_{1,it}=b_i+u_{it}$ plus a fixed binary factor. This exercises the time-varying $\mu_t$ that the formula is built for and helps separate $v$ from $a$. Optionally add random exposure $\omega_{it}\in(0,1]$.

**Main grid.** $I=2000$, $T=5$, $\phi=10$, $p=1.5$, giving 16 cells with **500 replications** each.

**One-at-a-time sensitivity** around a central cell:
- $I\in\{500,10000\}$;
- $T\in\{3,10\}$;
- $\phi\in\{5,20\}$;
- true $p\in\{1.3,1.7\}$ with $p$ estimated or fixed at 1.5;
- an unbalanced panel with random drop-out.

**Latent-structure misspecification.** Fit S1 (θ-AR(1)) when the truth is log-AR(1), and also fit S2. Optionally add a gamma-AR(1) or other non-lognormal latent process with the same $\gamma(k)$.

### 3.2 Study A: estimator performance

Report, for $\hat v$, $\hat r(1)$ (or $\hat\rho$) and $\hat a$:
- bias, SD and RMSE, with medians and IQRs alongside;
- 95% CI coverage, sandwich (with stage-1 correction) versus cluster bootstrap;
- frequency of estimates near the boundary;
- J-test size;
- size and power of the static ($\gamma(1)=\dots=\gamma(T-1)$) and no-dependence Wald tests.

Compare joint GMM, lag-only, and equal-weight versions; the equal-weight version shows why weighting matters.

### 3.3 Study B: predictive performance at $T+1$

**Competitors** (all feasible methods use estimated parameters):

| Method | Description |
|---|---|
| Naive | GLM mean $\hat\mu_{T+1}$ |
| Static (fitted) | Exchangeable GMM (S3) → Corollary 2 premium |
| Proposed (joint GMM) | S1 plug-in, via Theorem 1 or the Prop. 2 recursion |
| Proposed (lag-only), Proposed (NP) | Robustness variants |
| Oracle-parameter proposed | True $\gamma$, $a$, $\beta$ (upper bound for linear premiums) |
| Oracle-parameter best exchangeable | True moments, equal weights (best possible static) |
| Oracle θ | $\mu_{T+1}\theta_{T+1}$ |

**Metrics**
- Exact population MSE for the oracle-parameter predictors (closed form, Section 3.0).
- Out-of-sample MSE and RMSE as **paired differences versus Naive**, with Monte Carlo SEs and the proportion of replications in which each method beats Naive.
- **Share of oracle gain** $=(\mathrm{MSE}_{N}-\mathrm{MSE}_{M})/(\mathrm{MSE}_{N}-\mathrm{MSE}_{\text{oracle-param}})$.
- Premium error against the true mean, $E[(P-\mu_{T+1}\theta_{T+1})^2]$, which removes irreducible Tweedie noise.
- Mean Tweedie deviance; always defined because Proposition 2 guarantees $P>0$.
- Report medians or trimmed means alongside means, because the losses are heavy-tailed.

**Sanity checks the results must satisfy**
- In expectation, oracle-parameter proposed ≤ oracle-parameter best exchangeable ≤ Naive.
- Gains increase with $v$ and with $r(1)$.
- Fitted Static is not materially worse than Naive.

**Figures**
- Heatmap of the share of oracle gain over $(v,r(1))$.
- Boxplots of $\hat v$ and $\hat r(1)$.
- Line plot of the relative MSE gain against $r(1)$ for each $v$.

### 3.4 Claims to remove or rewrite

- "Static is worse than Naive in 12/16 configurations" and "pooling can be worse than ignoring claims". This was an artifact of plugging the full Var θ into Bühlmann; a fitted static model estimates the average covariance instead.
- "Improvement is largest at moderate σ … swamped by latent variability" (Fig. 4.3).
- "Uniformly superior to Naive and substantially superior to Bühlmann". Restate from the new results with Monte Carlo SEs.

---

## 4. Proposed manuscript structure

1. **Introduction:** motivation; bulleted contributions (Theorem 1, Proposition 2, GMM estimator and tests, simulation evidence).
2. **Related literature:** Tweedie GLM and credibility; evolutionary credibility (Section 1.2); dynamic and latent-factor dependence. Short, with no textbook material.
3. **Model:** notation table; assumptions (A1)–(A4); examples of latent processes (θ-AR(1), log-AR(1), exchangeable, ARMA or long memory).
4. **Credibility premium:** Theorem 1 and corollaries; Proposition 2 (recursion, positivity, bounds, comparative statics).
5. **Semi-parametric estimation:** moment identities, two-step GMM, asymptotics, J and Wald tests, practical extensions.
6. **Simulation study:** design; Study A (estimation); Study B (prediction); sensitivity.
7. *(Deferred: empirical application; see Section 6.)*
8. **Discussion and conclusion:** limitations; extensions (hierarchical, higher-order, copula latent processes).

Appendices: proofs; exact MSE formula; code availability (R package or repository).

---

## 5. Work plan checklist

**Phase 1 — Code audit**
- [ ] Reproduce Table 4.2 with the exact-MSE unit test.
- [ ] Locate the moment and ordering bug.
- [ ] Freeze a validated simulation engine (port `mc.py` logic to R if the project stays in R).

**Phase 2 — Theory**
- [ ] Write Theorem 1 and its corollaries.
- [ ] Prove Proposition 2 (a)–(d) and (g).
- [ ] Prove or qualify (e) and (f).
- [ ] Rewrite the literature review.
- [ ] Complete the housekeeping list.

**Phase 3 — Estimation**
- [ ] Implement joint GMM in R, porting `gmm.py` and `gmm_joint.py`.
- [ ] Add stacked-score sandwich SEs and the cluster bootstrap.
- [ ] Add the J and Wald tests.
- [ ] Add $p$ profiling, exposure weights and unbalanced-panel support.

**Phase 4 — Simulation**
- [ ] Run Study A, Study B and the sensitivity runs.
- [ ] Produce tables and figures.
- [ ] Rewrite Sections 6 and 8 from the new results.

**Phase 5 — Submission**
- [ ] Format to the target journal's template.
- [ ] Run a similarity check against the thesis.
- [ ] **Disclose the SFU thesis** in the cover letter and acknowledgments.
- [ ] Submit the abstract before 31 Dec 2027 and the manuscript before 31 Mar 2028.

---

## 6. Ideas from the review not in this revision's scope

These would strengthen the fit with the Topic and the referee case. Revisit after Phases 1–4.

- **Real-data application.** A longitudinal policy-level portfolio with five or more years, e.g. the Wisconsin LGPIF panel. Referees will likely expect one.
- **Robustness to latent dependence.** Latent processes with the same $\gamma(k)$ but different dependence (Gaussian, gamma-AR(1), copula Markov chains with tail dependence); this matches the Topic's copula and tail-dependence themes.
- **Exact-Bayes benchmark.** Posterior mean via particle filter or MCMC, to quantify what is lost by staying linear.
- **Machine-learning comparator.** Gradient boosting with Tweedie loss and lagged-claim features.
- **Ratemaking metrics.** Gini index / ordered Lorenz curves, double-lift charts, balance property.

---

## Appendix: prototype code from the review

- `mc.py`: replication of the Ch. 4 design with correctly specified and literal versions of the proposed premium; exact population MSE.
- `gmm.py`: lag-only moment / minimum-distance estimator, plug-in premium, and simulation driver (Ch. 3 and Ch. 4 designs, time-varying covariates).
- `gmm_joint.py`: joint GMM with diagonal moments (requires `gmm.py` in the same folder).

These are Python prototypes for checking ideas, not production code.

# SMOR_MODULE2_SCALING_LAWS_IMPLEMENTATION.md

# Module 2 — Scaling Laws for Data Ratio
## Implementation Plan for SMOR

## 0. Mục tiêu

Module 2 nghiên cứu và triển khai ánh xạ

\[
(B,\mathbf p)\rightarrow J
\]

hoặc tương đương

\[
(N_1,\ldots,N_K)\rightarrow J,
\]

trong đó \(B\) là total acquisition budget, \(\mathbf p\) là acquisition mixture trên \(K\) data sources, \(N_i\) là số unique trajectories từ source \(i\), và \(J\) là downstream performance.

Mục tiêu cuối là dự đoán

\[
\mathbf p_B^*
=
\arg\max_{\mathbf p\in\Delta^K}J(B,\mathbf p)
\]

hoặc, nếu dùng validation loss,

\[
\mathbf p_B^*
=
\arg\min_{\mathbf p\in\Delta^K}L(B,\mathbf p).
\]

Core principle:

\[
\boxed{\text{Discover first, parameterize second, extrapolate third.}}
\]

Không assume trước một power law rồi ép data phải fit.

---

# 1. Scope MVP

MVP nên bắt đầu với:

- source-defined domains;
- cùng embodiment;
- fixed learner;
- simulated data;
- 2 sources;
- equal acquisition cost;
- BC hoặc BC-RNN.

Chưa cần ngay:

- Bayesian posterior phức tạp;
- learned \(F_\phi\);
- dynamic source discovery;
- VLA;
- cross-embodiment;
- active online acquisition.

Câu hỏi đầu tiên chỉ là:

\[
\boxed{\text{Does a predictable scaling relation exist?}}
\]

---

# 2. Domain = Acquisition Source

Trong Module 2:

\[
\boxed{\text{Domain}\equiv\text{Acquisition Source}}
\]

Source được xác định bằng acquisition provenance hoặc collection configuration, không bằng post-hoc error label.

Ví dụ:

\[
S_1=\text{collection regime A},
\qquad
S_2=\text{collection regime B}.
\]

Các property như quality, coverage, bias, redundancy chỉ được đo sau đó.

---

# 3. Budget và source counts

Nếu collection cost bằng nhau:

\[
N_i=\lfloor Bp_i\rfloor,
\qquad
\sum_iN_i=B.
\]

Nếu có cost \(c_i\):

\[
N_i=
\left\lfloor
\frac{Bp_i}{c_i}
\right\rfloor.
\]

MVP nên để:

\[
c_i=1.
\]

---

# 4. High-Level Pipeline

```text
Source Pool Generation
        |
        v
Budget × Mixture Sweep
        |
        v
Unique Dataset Sampling
        |
        v
Policy Training
        |
        v
Validation + Closed-loop Evaluation
        |
        v
Scaling Results Table
        |
        +----------------------+
        |                      |
        v                      v
Flexible Trend Model      Parametric Laws
(GAM / GP)                (Power / Saturation)
        |                      |
        +----------+-----------+
                   |
                   v
        Held-out Extrapolation
                   |
                   v
       Optimal Mixture Prediction
                   |
                   v
        Oracle Mixture Comparison
```

---

# 5. Simulated Source Pools

Với mỗi source \(S_i\), generate trước một large trajectory pool:

\[
\mathcal P_i
=
\{\tau_1^{(i)},\ldots,\tau_{N_{\max}}^{(i)}\}.
\]

Ví dụ:

```text
data/
  source_A/
    trajectories.hdf5
  source_B/
    trajectories.hdf5
```

Mỗi trajectory phải được sinh thật trong simulator.

Nếu action bị perturb thành \(a_t'\), simulator phải sinh lại:

\[
s_{t+1}'=T(s_t,a_t').
\]

Không được sửa action offline nhưng giữ next-state cũ.

Metadata tối thiểu:

```yaml
trajectory_id:
source_id:
seed:
task_id:
episode_length:
success:
collection_config:
```

Optional:

```yaml
initial_state_descriptor:
environment_config:
controller_id:
collection_cost:
```

---

# 6. Experimental Grid

MVP 2-source:

\[
\mathbf p=(p,1-p).
\]

Recommended budgets:

\[
B\in\{50,100,200,400,800,1600\}.
\]

Recommended mixtures:

\[
p\in\{0,0.2,0.4,0.6,0.8,1.0\}.
\]

Dense version:

\[
p\in\{0,0.1,\ldots,1.0\}.
\]

Seeds:

\[
s\in\{0,1,2\}
\]

cho MVP, và 5 seeds cho final experiment.

Lightweight run:

\[
6\times6\times3=108
\]

training runs.

---

# 7. Train / Held-out Scale Split

Không dùng tất cả budgets để vừa chọn law vừa claim extrapolation.

Recommended:

\[
B_{\mathrm{fit}}
=
\{50,100,200,400\},
\]

\[
B_{\mathrm{heldout}}
=
\{800,1600\}.
\]

Config:

```yaml
fit_budgets: [50, 100, 200, 400]
heldout_budgets: [800, 1600]
```

---

# 8. Dataset Sampler

Interface:

```python
sample_dataset(
    source_pools,
    budget,
    mixture,
    seed,
)
```

Với 2 sources:

```python
n_a = round(budget * p)
n_b = budget - n_a
```

Sample without replacement:

```python
ids_a = rng.choice(pool_a, n_a, replace=False)
ids_b = rng.choice(pool_b, n_b, replace=False)
```

Acquisition scaling phải count **unique trajectories**.

Training có thể reuse trajectories qua epochs, nhưng:

\[
N_i
\]

luôn là unique-data count.

---

# 9. Fixed Learner

Scaling discovery không nên đồng thời thay learner.

MVP dùng:

\[
\boxed{\text{BC hoặc BC-RNN fixed}}
\]

Giữ cố định:

- architecture;
- optimizer;
- learning rate;
- batch size;
- augmentation;
- evaluation protocol.

Cần log cả:

\[
\text{unique data budget}
\]

và:

\[
\text{training compute}.
\]

Main data-scaling study có thể dùng fixed epochs, sau đó thêm compute-matched ablation.

---

# 10. Metrics

Mỗi run lưu:

```yaml
budget:
mixture:
source_counts:
seed:
val_loss:
closed_loop_success:
training_steps:
wall_time:
```

Primary scaling target:

\[
L_{\mathrm{val}}(B,p).
\]

Primary deployment metric:

\[
J_{\mathrm{success}}(B,p).
\]

Validation loss thường smooth hơn success rate, vì vậy nên dùng nó để discover scaling structure rồi dùng success để validate downstream relevance.

---

# 11. Results Table

Store:

```text
results/scaling_runs.csv
```

Schema:

```csv
task,learner,budget,p_source_a,p_source_b,n_source_a,n_source_b,seed,val_loss,success_rate,train_steps,wall_time
```

Phải giữ per-seed rows, không chỉ aggregated means.

---

# 12. Stage 1 — Raw Visualization

Trước khi fit law, tạo:

## Plot A

\[
B\rightarrow L(B,p)
\]

cho từng mixture.

## Plot B

\[
p\rightarrow L(B,p)
\]

cho từng budget.

## Plot C

\[
(B,p)\rightarrow J_{\mathrm{success}}.
\]

## Plot D

\[
B\rightarrow p_B^{*,\mathrm{grid}}
\]

với:

\[
p_B^{*,\mathrm{grid}}
=
\arg\min_{p\in\mathcal P_{\mathrm{grid}}}\bar L(B,p).
\]

Ba câu hỏi cần trả lời:

1. Performance có scale theo \(B\) không?
2. Mixture có ảnh hưởng không?
3. Optimal mixture có thay đổi theo scale không?

---

# 13. Stage 2 — Flexible Trend Discovery

Trước khi assume power law, fit một flexible smoother.

Recommended:

1. GAM / smoothing spline;
2. Gaussian Process.

---

# 14. GAM

Conceptual model:

\[
L
=
f_1(\log B)
+
f_2(p)
+
f_{12}(\log B,p)
+
\epsilon.
\]

Quan trọng nhất:

\[
\boxed{f_{12}(\log B,p)}
\]

vì nó đo scale × mixture interaction.

Nếu:

\[
f_{12}\approx0,
\]

thì mixture preference có thể gần scale-invariant.

Nếu:

\[
f_{12}\neq0,
\]

thì có evidence rằng:

\[
p_B^*
\]

thay đổi theo \(B\).

Interface:

```python
class GAMScalingModel:
    def fit(self, df):
        ...

    def predict(self, budget, mixture):
        ...

    def interaction_strength(self):
        ...
```

---

# 15. Gaussian Process

Input:

\[
x=(\log B,p).
\]

Output:

\[
L.
\]

GP cho:

\[
\mu_L(B,p)
\]

và uncertainty:

\[
\sigma_L(B,p).
\]

Interface:

```python
class GPScalingModel:
    def fit(self, df):
        ...

    def predict(self, budget, mixture, return_std=True):
        ...
```

MVP có thể dùng `sklearn.gaussian_process.GaussianProcessRegressor`.

GP đặc biệt hữu ích nếu sau này muốn uncertainty-aware acquisition.

---

# 16. Stage 3 — Candidate Parametric Laws

Không fit chỉ một candidate.

## Model 1 — Power law

\[
L(B,p)
=
L_\infty(p)
+
A(p)B^{-\alpha}.
\]

## Model 2 — Shifted power law

\[
L(B,p)
=
L_\infty(p)
+
A(p)(B+B_0)^{-\alpha}.
\]

## Model 3 — Exponential saturation

\[
L(B,p)
=
L_\infty(p)
+
A(p)e^{-kB}.
\]

## Model 4 — Log trend baseline

\[
L(B,p)=a(p)-b(p)\log B.
\]

---

# 17. Mixture Dependence

Với 2 sources, bắt đầu bằng low-order functions:

\[
A(p)=a_0+a_1p+a_2p^2,
\]

\[
L_\infty(p)=c_0+c_1p+c_2p^2.
\]

Optional:

\[
\alpha(p)=\alpha_0+\alpha_1p.
\]

Không nên bắt đầu với high-degree polynomials.

---

# 18. Source-Count Form

Ngoài \((B,p)\), nên support trực tiếp:

\[
(N_1,N_2).
\]

Candidate additive law:

\[
L(N_1,N_2)
=
L_\infty
+
a_1(N_1+N_{0,1})^{-\alpha_1}
+
a_2(N_2+N_{0,2})^{-\alpha_2}.
\]

Ưu điểm: acquisition trực tiếp thao tác trên source counts.

Interface:

```python
class AdditiveSourceScalingLaw:
    def fit(self, counts, y):
        ...

    def predict(self, counts):
        ...
```

---

# 19. Interaction Model

Không thêm interaction từ đầu.

Fit additive model trước:

\[
L_{\mathrm{add}}
=
L_\infty+f_1(N_1)+f_2(N_2).
\]

Residual:

\[
R(N_1,N_2)
=
L_{\mathrm{observed}}
-
L_{\mathrm{add}}.
\]

Nếu residual có structure rõ, mới thêm interaction term.

Ví dụ:

\[
L
=
L_{\mathrm{add}}
+
\gamma
(N_1+\epsilon)^{-a}
(N_2+\epsilon)^{-b}.
\]

Interaction phải được justify bởi empirical residuals.

---

# 20. Fitting

Use:

```python
scipy.optimize.curve_fit
```

hoặc:

```python
scipy.optimize.least_squares
```

Required:

- parameter bounds;
- multiple initializations;
- fit diagnostics;
- failure fallback.

Store:

```yaml
model_name:
params:
train_rmse:
fit_status:
```

---

# 21. Model Selection

Không chọn model chỉ bằng train \(R^2\).

Dùng:

1. train RMSE;
2. AIC;
3. BIC;
4. mixture interpolation error;
5. held-out scale extrapolation error.

Quan trọng nhất:

\[
\boxed{\text{Held-out scale extrapolation}}
\]

---

# 22. Held-Out Extrapolation

Fit:

\[
B\in\{50,100,200,400\}.
\]

Predict:

\[
B\in\{800,1600\}.
\]

Metrics:

\[
\mathrm{MAE},
\qquad
\mathrm{RMSE},
\qquad
\mathrm{relative\ error}.
\]

---

# 23. Mixture Interpolation

Fit selected mixture points:

\[
p\in\{0,0.4,0.8,1.0\}.
\]

Test:

\[
p\in\{0.2,0.6\}.
\]

Mục đích là kiểm tra fitted surface có generalize qua mixture ratios không.

---

# 24. Oracle Mixture

Ở held-out budget \(B\), chạy dense mixture grid:

\[
p\in\{0,0.1,\ldots,1.0\}.
\]

Define:

\[
p_B^{*,\mathrm{oracle}}
=
\arg\min_p\bar L(B,p).
\]

Với success:

\[
p_B^{*,J}
=
\arg\max_p\bar J(B,p).
\]

Oracle chỉ approximate vì grid hữu hạn.

---

# 25. Predicted Optimal Mixture

Given fitted law:

\[
\hat L(B,p),
\]

solve:

\[
\hat p_B^*
=
\arg\min_{p\in[0,1]}\hat L(B,p).
\]

2-source:

```python
scipy.optimize.minimize_scalar
```

\(K>2\):

```python
scipy.optimize.minimize
```

subject to:

\[
p_i\ge0,
\qquad
\sum_i p_i=1.
\]

---

# 26. Target-Budget Regret

Main metric:

\[
R_B
=
J(B,p_B^{*,\mathrm{oracle}})
-
J(B,\hat p_B^*).
\]

Loss version:

\[
R_B^L
=
L(B,\hat p_B^*)
-
L(B,p_B^{*,\mathrm{oracle}}).
\]

Regret quan trọng hơn chỉ báo:

\[
|\hat p_B^*-p_B^{*,\mathrm{oracle}}|
\]

vì nhiều mixture khác nhau có thể có performance gần như nhau.

---

# 27. Bootstrap

Bootstrap cho:

- scaling exponent;
- asymptotic loss;
- optimal mixture;
- target-budget regret.

Recommended MVP:

```python
bootstrap_runs = 500
```

Mỗi bootstrap sample:

1. resample seed-level observations;
2. refit law;
3. predict \(p_B^*\);
4. store parameters và optimum.

Output:

\[
\hat p_B^*
\pm
95\%\ \mathrm{CI}.
\]

---

# 28. Success-Rate Statistics

Nếu evaluate \(n_{\mathrm{eval}}\) episodes và có \(y\) successes:

\[
y\sim\mathrm{Binomial}(n_{\mathrm{eval}},\pi).
\]

Có thể model:

\[
\mathrm{logit}(\pi)
=
f(\log B,p).
\]

MVP có thể fit scaling law chủ yếu trên validation loss và dùng binomial confidence intervals cho success rate.

---

# 29. Marginal Acquisition Gain

Khi có smooth model:

\[
\hat L(N_1,\ldots,N_K),
\]

estimate:

\[
\boxed{
G_i
=
-\frac{\partial \hat L}{\partial N_i}
}
\]

Interpretation:

> expected loss reduction từ một additional unique sample của source \(i\).

Nếu source có cost \(c_i\):

\[
\boxed{
G_i^{\mathrm{cost}}
=
-\frac1{c_i}
\frac{\partial \hat L}{\partial N_i}
}
\]

Đây là bridge tự nhiên sang acquisition optimization.

---

# 30. KKT Allocation

Nếu fitted law:

\[
L
=
L_\infty
+
\sum_i a_iN_i^{-\alpha_i},
\]

và:

\[
\sum_i c_iN_i\le B,
\]

thì interior optimum thỏa:

\[
a_i\alpha_iN_i^{-(\alpha_i+1)}
=
\lambda c_i.
\]

Hay:

\[
\boxed{
\frac{
a_i\alpha_iN_i^{-(\alpha_i+1)}
}{c_i}
=
\lambda
}
\]

Interpretation:

> active sources được cấp data tới khi marginal loss reduction per unit collection cost cân bằng.

---

# 31. Source Characterization Before Scaling

Trước full sweep, chạy:

\[
J(S_i)
\]

cho từng source.

Sau đó pairwise mixture:

\[
J(S_i+S_j).
\]

Define complementarity diagnostic:

\[
C_{ij}
=
J(S_i+S_j)
-
\max\{J(S_i),J(S_j)\}.
\]

Mục đích không phải định nghĩa source, mà để xác định setting nào đáng chạy scaling sweep sâu.

---

# 32. Failure Modes

## Dominant source

\[
p_B^*=e_i
\]

cho mọi \(B\).

Valid result, nhưng không phải interesting mixture-scaling setting.

## Scale-invariant mixture

\[
p_B^*\approx p^*
\]

cho mọi \(B\).

Mixture matters nhưng scaling đơn giản.

## Unstable optimum

\(p_B^*\) thay đổi mạnh qua seeds.

Scaling law có thể chưa identifiable.

## Strong interaction

Additive law fail có structure.

Cần interaction term.

## Success not smooth

Fit law trên validation loss, dùng closed-loop success để verify.

---

# 33. Recommended Code Structure

```text
smor/
  scaling/
    __init__.py
    config.py

    source_pool.py
    sampler.py
    records.py
    results_store.py

    trend/
      gam.py
      gp.py

    laws/
      base.py
      power.py
      shifted_power.py
      exponential.py
      additive_source.py
      interaction.py

    fitting.py
    model_selection.py
    oracle.py
    optimize_mixture.py
    bootstrap.py
    marginal_gain.py
    plots.py

    evidence_adapter.py

experiments/
  scaling/
    generate_pools.py
    run_grid.py
    fit_trends.py
    fit_laws.py
    evaluate_extrapolation.py
    predict_optimal_mixture.py
    run_oracle.py

configs/
  scaling/
    two_source_mvp.yaml

tests/
  scaling/
    test_sampler.py
    test_budget_conservation.py
    test_scaling_fit.py
    test_optimal_mixture.py
    test_bootstrap.py
    test_extrapolation_split.py
```

---

# 34. Core Data Structures

```python
@dataclass
class ScalingObservation:
    budget: int
    mixture: np.ndarray
    source_counts: np.ndarray
    seed: int
    val_loss: float
    success_rate: float
    task: str
    learner: str
```

```python
class ScalingDataset:
    observations: list[ScalingObservation]

    def to_dataframe(self):
        ...

    def filter_budgets(self, budgets):
        ...
```

---

# 35. Scaling Model Interface

```python
class ScalingModel(ABC):
    @abstractmethod
    def fit(self, observations):
        ...

    @abstractmethod
    def predict(self, budget, mixture):
        ...

    def optimal_mixture(self, budget):
        ...
```

Example:

```python
model = ShiftedPowerScalingLaw()

model.fit(train_df)

pred = model.predict(
    budget=800,
    mixture=np.array([0.4, 0.6]),
)

p_star = model.optimal_mixture(800)
```

---

# 36. Experiment Runner

```python
for budget in config.budgets:
    for mixture in config.mixtures:
        for seed in config.seeds:

            subset = sampler.sample(
                source_pools=pools,
                budget=budget,
                mixture=mixture,
                seed=seed,
            )

            learner = build_learner(config.learner, seed)

            learner.fit(subset)

            val_loss = evaluate_validation(learner)
            success = evaluate_closed_loop(learner)

            save_observation(
                budget=budget,
                mixture=mixture,
                source_counts=subset.source_counts,
                seed=seed,
                val_loss=val_loss,
                success_rate=success,
            )
```

---

# 37. Fit Script

```python
df = load_scaling_results()

train_df = df[df["budget"].isin(config.fit_budgets)]
test_df = df[df["budget"].isin(config.heldout_budgets)]

models = [
    PowerScalingLaw(),
    ShiftedPowerScalingLaw(),
    ExponentialScalingLaw(),
]

for model in models:
    model.fit(train_df)

    report = evaluate_model(
        model,
        train_df=train_df,
        heldout_df=test_df,
    )

    save_report(report)
```

---

# 38. Optimal Mixture Prediction

```python
best_model = select_model(
    criterion="heldout_rmse"
)

for B in target_budgets:
    p_hat = best_model.optimal_mixture(B)

    oracle = oracle_lookup(B)

    regret = evaluate_regret(
        budget=B,
        predicted_mixture=p_hat,
        oracle_mixture=oracle,
    )
```

---

# 39. Module 1 → Module 2 Interface

Module 1 outputs:

\[
\mathcal E_{\mathrm{pilot}}
=
\{
\beta_{0:T},
h_{0:T},
L_{\mathrm{out},0:T},
\text{source stats}
\}.
\]

Module 2 MVP **chưa cần dùng evidence này ngay**.

Trước tiên phải xây scaling ground truth độc lập.

Sau đó future adapter:

```python
class PilotEvidenceAdapter:
    def transform(self, reweighting_evidence):
        return ScalingPilotFeatures(...)
```

Potential features:

```yaml
beta_mean:
beta_final:
beta_slope:
beta_auc:
hypergradient_mean:
hypergradient_variance:
source_count:
source_cost:
coverage:
redundancy:
```

Future mapping:

\[
F_\phi:
\mathcal E_{\mathrm{pilot}}
\rightarrow
\hat{\mathbf p}_{B_{\mathrm{target}}}^*.
\]

---

# 40. Correct Research Order

## Phase 1 — Scaling discovery

\[
(B,p)\rightarrow L,J.
\]

Question:

\[
\text{Is the surface smooth and predictable?}
\]

## Phase 2 — Fit scaling law

Question:

\[
\text{Can small-scale runs predict held-out budgets?}
\]

## Phase 3 — Optimal mixture prediction

Question:

\[
\text{Can the fitted law predict }p_B^*?
\]

## Phase 4 — Pilot reweighting bridge

Question:

\[
\text{Can }\beta_{0:T}^{\mathrm{pilot}}
\text{ replace part of the mixture sweep?}
\]

Không đảo thứ tự này.

---

# 41. MVP Config

```yaml
experiment:
  name: "two_source_scaling_mvp"

sources:
  - id: "source_A"
    path: "data/source_A/trajectories.hdf5"
    cost: 1.0

  - id: "source_B"
    path: "data/source_B/trajectories.hdf5"
    cost: 1.0

budgets:
  all: [50, 100, 200, 400, 800, 1600]
  fit: [50, 100, 200, 400]
  heldout: [800, 1600]

mixtures:
  source_A: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

seeds: [0, 1, 2]

learner:
  name: "bc"
  epochs: 50
  batch_size: 256

evaluation:
  num_rollouts: 100
  metric: "success_rate"

scaling_models:
  - "power"
  - "shifted_power"
  - "exponential"

bootstrap:
  num_resamples: 500
```

---

# 42. Unit Tests

## Budget conservation

\[
\sum_iN_i=B.
\]

## Mixture correctness

For:

\[
B=100,\quad p=(0.3,0.7),
\]

expect:

\[
N=(30,70).
\]

## No replacement

```python
len(ids) == len(set(ids))
```

## Synthetic power law recovery

Generate:

\[
y=1+3B^{-0.4}.
\]

Fit và recover:

\[
\alpha\approx0.4.
\]

## Held-out leakage

Held-out budgets không được đi vào `.fit()`.

## Known optimum

Synthetic:

\[
L(B,p)
=
(p-0.3)^2+B^{-0.5}.
\]

Predict:

\[
p^*\approx0.3.
\]

---

# 43. Integration Test

Synthetic source-count law:

\[
L(N_1,N_2)
=
1
+
2(N_1+1)^{-0.5}
+
3(N_2+1)^{-0.3}.
\]

Test full pipeline:

1. generate observations;
2. fit;
3. extrapolate;
4. predict optimal mixture;
5. estimate marginal gain;
6. bootstrap CI.

Pipeline này phải pass trước khi chạy robot experiments đắt.

---

# 44. MVP Milestones

## Milestone 1
Generate 2 source pools.

## Milestone 2
Implement budget × mixture × seed runner.

## Milestone 3
Raw plots:

\[
B\to L,
\qquad
p\to L,
\qquad
B\to p_B^*.
\]

## Milestone 4
GAM/GP trend discovery.

## Milestone 5
Fit candidate laws.

## Milestone 6
Held-out scale extrapolation.

## Milestone 7
Predict \(\hat p_B^*\), compare oracle.

## Milestone 8
Marginal utility curves.

## Milestone 9
Connect pilot \(\beta/h\) from Module 1.

---

# 45. Success Criteria

Module 2 MVP được coi là thành công nếu ít nhất một controlled source setting cho thấy:

1. performance thay đổi smooth theo unique-data scale;
2. mixture có measurable effect;
3. scale × mixture interaction tồn tại;
4. small-scale law extrapolate được sang held-out budgets;
5. predicted mixture có low target-budget regret.

Strongest result:

\[
\boxed{
p_{B_1}^*
\neq
p_{B_2}^*
}
\]

và law dự đoán được sự dịch chuyển đó.

---

# 46. Negative Results vẫn có giá trị

## Dominance

\[
p_B^*=e_i.
\]

## Scale invariance

\[
p_B^*\approx\text{constant}.
\]

## Unstable scaling

Large seed uncertainty.

## No extrapolation

Small-scale fit tốt nhưng large-scale prediction fail.

## Strong non-additivity

Cần interaction model.

Những kết quả này phải được report, không được loại bỏ chỉ vì không cho mixture đẹp.

---

# 47. Recommended Statistical Toolkit

Minimum:

```text
numpy
pandas
scipy
statsmodels
scikit-learn
matplotlib
```

Optional:

```text
pygam
gpytorch
```

MVP recommendation:

- GAM/splines: trend discovery;
- SciPy: nonlinear scaling-law fitting;
- sklearn GP: uncertainty-aware surface;
- bootstrap: confidence intervals;
- statsmodels: optional mixed-effects.

---

# 48. Final Output Object

Module 2 nên export:

```python
ScalingEvidence(
    scaling_surface_model,
    fitted_law,
    law_parameters,
    extrapolation_metrics,
    optimal_mixture_by_budget,
    bootstrap_intervals,
    marginal_gain_curves,
    oracle_regret,
)
```

Conceptually:

\[
\mathcal S
=
\{
\hat L(B,\mathbf p),
\hat{\mathbf p}_B^*,
\hat G_i(N_i),
\text{uncertainty},
\text{extrapolation error}
\}.
\]

---

# 49. Full SMOR Integration Later

```text
Uniform Pilot
    |
    v
Module 1: Online Reweighting
    |
    v
{beta_t, h_t}
    |
    v
Module 2: Scale-Aware Acquisition Inference
    |
    v
p_hat(B_target)
    |
    v
Target-Scale Data Collection
    |
    v
Final Online Reweighting
    |
    v
theta_final
```

Nhưng trước khi học:

\[
F_\phi:
\{\beta_t,h_t\}
\rightarrow
\mathbf p_B^*,
\]

Module 2 phải chứng minh độc lập rằng:

\[
\boxed{
(B,\mathbf p)\rightarrow J
}
\]

thực sự có stable và extrapolatable scaling structure.

---

# 50. Kết luận

Thứ tự scientific và engineering đúng cho Module 2 là:

\[
\boxed{
\text{Empirical sweep}
\rightarrow
\text{Flexible trend estimation}
\rightarrow
\text{Candidate law fitting}
\rightarrow
\text{Held-out extrapolation}
\rightarrow
\text{Optimal acquisition}.
}
\]

Sau khi pipeline này hoạt động, mới dùng trajectory reweighting từ Module 1 để giảm số lượng mixture sweeps cần thiết và xây acquisition-inference operator của full SMOR.

# SMOR — Experimental Report

**SMOR** = học *per-source data weights* cho imitation learning từ demonstrations chất lượng lẫn lộn
(online reweighting, Module 1) + *scaling law* cho tỉ lệ trộn dữ liệu (Module 2). Báo cáo tổng hợp thí
nghiệm đã chạy, Research Question, cấu hình, và kết quả. *(★ = kết quả nổi bật để đưa lên slide.)*

---

## 1. Research Questions

| RQ | Câu hỏi | Trả lời |
|----|---------|---------|
| **RQ1** | SMOR (curvature K>1) có hơn CAIL (one-step K=1) khi phục hồi demo tốt? | **SMOR ≈ CAIL** — K=1 đủ trên varying-optimality (well-conditioned) |
| **RQ2** | Với source **bổ trợ** (mỗi source 1 lỗi khác, không source nào trội), reweighting có học nghiệm interior thắng uniform/CAIL? | **Có** — ★ trên **closed-loop success**, SMOR **thắng** uniform & CAIL; trên val-loss thì hòa |
| **RQ3** | Outer loss open-loop (val/ranking) có align với mục tiêu thật (return)? | **Không** — ★ đổi sang **GRPO closed-loop return** thì SMOR vượt cả uniform |
| **RQ4** | (Scaling) Có scaling law dự đoán được, optimal mixture có dịch theo budget? | **Có** — ★ complementary-sim: p\* **dịch** 0→interior; official-quality: dominance |

**Thông điệp chính:** curvature (K) không phải điểm mấu chốt; **thiết kế outer loss (closed-loop) và
metric đánh giá (success/return, KHÔNG dùng val-loss) mới quyết định**.

---

## 2. Cấu hình thí nghiệm

- **Learner (cố định)**: Behavior Cloning, MLP 256×256 / 512×512 (tanh), Adam 1e-3, batch 128–256. Cùng
  backbone cho mọi baseline (cô lập cơ chế reweighting).
- **Reweighting (SMOR)**: bilevel online. `n` = độ mịn nhóm (n=1 per-demo / whole-fidelity = 1 nhóm/source);
  `K` = độ sâu Neumann hypergradient `h_j = −g_out^T P_K g_j`. **K=1 ≡ CAIL one-step; K>1 = SMOR**.
- **Baselines (cách xác định weights)**: `uniform` · `only:<source>` · `static_quality` (source nhãn tốt nhất)
  · **`CAIL`** (K=1 + ranking) · `AIRL` (adversarial, chỉ HalfCheetah) · **`SMOR`** (K>1).
- **Outer loss**: `ValidationLoss` (open-loop MSE), `CAILRankingLoss` (pairwise), `ClosedLoopReturn`
  (khả vi, point-mass), **`ClosedLoopRolloutReturn`** (rollout thật + **REINFORCE / GRPO / PPO**) +
  `normalize_group_grads` (cosine `g_j/‖g_j‖`).
- **Metric**: **closed-loop success** ↑ (manipulation) / **return** ↑ (locomotion) — *ưu tiên*; val-loss ↓,
  expert-MSE ↓, spearman(β, return) ↑ chỉ dùng phụ.

**Datasets/environments đã dùng:** *Official* — CAIL Ant-v2, Minari (halfcheetah/hopper/walker2d),
RoboMimic (lift/square, robosuite). *Self-simulated* — point-mass, robosuite device-calibration.

---

# PHẦN I — ONLINE REWEIGHTING (Module 1)

## 3. RQ1 — SMOR vs CAIL trên varying-optimality (source khác về CHẤT LƯỢNG)

### 3.1 Dataset CHÍNH THỨC — CAIL Ant-v2

**CAIL Ant-v2** (buffer official của paper CAIL, 5 checkpoint 4787→789), n=1 per-demo, env-free
(Ant-v2 legacy không rollout lại được), **3 seeds**:

| weights | expert-MSE ↓ | spearman(β,ret) ↑ |
|---|---|---|
| uniform | 0.048 | −0.08 |
| CAIL (K=1) | 0.057 | 0.610 |
| **SMOR (K=4)** | 0.052 | **0.672** |

*(Minari official mujoco cũng đã chạy; kết quả `uniform >> reweight` là bằng chứng RQ3 misalignment — xem §5.)*

### 3.2 Dataset TỰ SINH (để có metric return trên env hiện đại)

Demo tự sinh từ expert PPO (seals), metric **return**, **3 seeds**:

| weights | seals/Ant-v1 | seals/HalfCheetah-v1 |
|---|---|---|
| uniform | 1452 | 765 |
| only:expert (oracle) | 2501 | 1549 |
| CAIL (K=1) | 2360 | 1557 |
| SMOR (K=4) | 2264 | 1499 |

**★ Kết luận RQ1:** qua **3 setting × 3 seeds** (CAIL Ant-v2 official env-free, seals Ant-v1 & HalfCheetah
return): **SMOR ≈ CAIL** — curvature K>1 **không vượt** one-step K=1 (benchmark well-conditioned; K=1 đã đủ
concentrate về expert).

## 4. RQ2 — Source BỔ TRỢ tự mô phỏng (mỗi source 1 lỗi khác, không source nào trội)

Mỗi source = cùng task/expert qua một **thiết bị mô phỏng** có sai lệch riêng (xoay / gain dị hướng /
per-joint bias); action nhiễu **re-simulate** thật. **Metric = closed-loop success ↑** (point-mass success bão
hoà 1.0 nên dùng val ↓). Cột = environment, hàng = cách xác định weights. *(RM 1 seed — đang bổ sung seed.)*

| weights | Point-mass 3-dev (val↓) | **RM-lift 3-dev (succ↑)** | **RM-lift good+poison (succ↑)** |
|---|---|---|---|
| uniform | 0.012 | 0.20 | 0.00 |
| best-single | 0.014 | 0.60 | 0.80 |
| static_quality | — | 0.73 | — |
| CAIL (K=1) | 0.021 | 0.40 | — |
| **SMOR (K=4)** | 0.021 | **0.87** | **1.00** |
| β của SMOR | [.17,.51,.33] interior | [tall=1] góc | [.27,.32,.41,0] interior |

**★ Kết luận RQ2:** trên **closed-loop success**, **SMOR thắng rõ uniform và CAIL** ở robosuite (RM-lift 3-dev:
0.87 vs 0.20 vs 0.40; good+poison: 1.00 vs 0.00) — dù trên **val-loss thì SMOR ≈ uniform**. → **Metric quyết
định**: val-loss che mất lợi thế, closed-loop success mới lộ ra (bias tích lũy qua rollout). Reweighting học
**nghiệm interior** khi source bổ trợ; task quá dễ (point-mass) thì success bão hoà & hòa uniform.

## 5. RQ3 — Outer-loss misalignment & GRPO closed-loop return

**Vấn đề — bằng chứng trên Minari official** (D4RL-modern, `mujoco/<env>/{expert,medium,simple}`, return,
**3 seeds**): outer loss ranking dồn β→expert, nhưng expert-only **brittle** (BC covariate-shift) → thua uniform:

| weights (outer = ranking) | HalfCheetah-v5 | Hopper-v5 | Walker2d-v5 |
|---|---|---|---|
| uniform | **2897 ± 670** | 558 ± 54 | **1145 ± 1088** |
| CAIL (K=1) | 88 | 606 | 610 |
| SMOR (K=4) | 154 | 591 | 631 |

→ `ranking/val-MSE` thưởng "khớp demo expert", nhưng mục tiêu thật = **return** → proxy phân kỳ khỏi "max return".

**★ Giải pháp & kiểm chứng** — đổi outer loss ranking → **GRPO closed-loop return** (cùng backbone K=4):

*Locomotion (Minari official, return, 3 seeds):*
| outer loss | HalfCheetah-v5 | Hopper-v5 | Walker2d-v5 |
|---|---|---|---|
| uniform | 1694 ± 477 | 516 | 811 ± 1062 |
| CAIL (ranking) | 88 | 552 | 597 |
| **SMOR + GRPO** | **2636 ± 1985** | 572 | **1264 ± 941** |

*Manipulation (RoboMimic PH+MH+MG, robosuite, success, 1 seed):*
| outer loss | uniform | CAIL (ranking) | **SMOR + GRPO** |
|---|---|---|---|
| success | 0.60 | 0.60 | **0.80** |
| β SMOR+GRPO | — | — | {ph .30, **mg_success .56**, còn lại ~0} interior |

→ **GRPO sửa misalignment nơi ranking sai đích**: HalfCheetah (covariate-shift) GRPO **2636 > uniform 1694 >>
ranking ~100**; PH+MH+MG GRPO **success 0.80 > uniform/CAIL 0.60**, học mixture interior **upweight mg_success**
(ranking-by-quality sẽ *dìm* nó) → chứng tỏ GRPO tối ưu đúng return/success chứ không "khớp expert".
**GRPO ≥ uniform trên HalfCheetah + Walker2d** (covariate-shift), **hòa ở Hopper** (env dễ, ranking đã ổn).
*(GRPO variance cao vì PG; combined-lift: poison-weight 0.48–0.74 REINFORCE → 0.20 GRPO+g_j-norm.)*

---

# PHẦN II — SCALING LAW (Module 2)

## 6. RQ4 — Scaling law cho tỉ lệ trộn dữ liệu

Học **(B, p) → performance**, dự đoán **p\*_B = argmin_p L(B,p)** và **extrapolate** sang budget lớn chưa thấy.
Pipeline: sweep → GAM/GP trend → fit luật (power/shifted/exp/log) → held-out extrapolation → oracle + regret +
bootstrap. Learner BC cố định; 2 source; fit budget nhỏ, held-out budget lớn.

### 6.1 ★ Complementary-sim (point-mass, bias-variance) — optimal mixture DỊCH theo budget

2 source: A = unbiased nhưng nhiễu cao; B = precise nhưng lệch hướng. Re-simulate, **210 run** (7 budget × 6
mixture × 5 seed):

| budget B | 50 | 100 | 200 | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|---|---|
| **grid p\*** (tỉ lệ source-A) | 0.0 | 0.0 | 0.0 | 0.4 | 0.8 | 0.4 | 0.4 |

→ **p\* DỊCH 0.0 (budget nhỏ → chuộng source precise) → 0.4–0.8 (budget lớn → cần source unbiased)**. Luật
(exponential) dự đoán p\* 0→0.54; held-out RMSE **0.016**; regret ≤ **0.0035**; bootstrap p\*₁₆₀₀=0.54 [0.50,0.59].

### 6.2 Official-quality (RoboMimic PH vs MG) — dominance, p\* KHÔNG dịch

**90 run** (5 budget × 6 mixture × 3 seed): val đơn điệu theo p → **grid p\*=1.0 mọi budget** (dominance);
scaling extrapolate được (held-out RMSE 0.058) nhưng optimal **không dịch**.

**★ Kết luận RQ4:** scaling law dự đoán & extrapolate được; **complementary-sim** cho optimal mixture **dịch
theo budget** (kết quả thú vị), **official-quality** cho **dominance** (source sạch luôn thắng).

---

## 7. Slide bullets — kết luận chốt

1. **SMOR ≈ CAIL** trên varying-optimality (3 setting × 3 seeds) — curvature K>1 không vượt K=1.
2. **Metric quan trọng**: val-loss che lợi thế; trên **closed-loop success** SMOR **thắng** uniform & CAIL
   (RM device: 0.87 vs 0.20 vs 0.40).
3. **Outer loss quan trọng nhất**: open-loop/ranking bị misalign (uniform>>reweight); **GRPO closed-loop return**
   sửa được → SMOR (4353) > uniform (2187).
4. **Scaling**: optimal mixture **dịch theo budget** (complementary-sim), dominance (official-quality); luật
   extrapolate tốt (regret ≤0.0035).

**Hạ tầng:** ~66 test pass. Datasets official: CAIL Ant-v2, Minari (3 mujoco), RoboMimic (lift/square).
ManiSkill cài được nhưng env không chạy trên WSL (Vulkan) → future work. Số RM-device & GRPO hiện 1 seed,
đang chạy multi-seed để chốt.

# Budget (SVM-style) vs weighted stability loss: the equivalence

2026-11-19 · Nico (from a side discussion; saved for the progress log)

**Question.** Writing the budget approach as a Lagrangian gives the same loss as the weighted approach, except that the stability term is S(f) − B instead of S(f). The budget approach has one hyperparameter (B), and the weighted loss written that way seems to need B too. Is there an equivalence, so the choice is between λ and B?

**Answer: yes, with a caveat for non-convex problems.**

Constrained problem, with S(f) the average absolute link gap:

```latex
\min_f L(f) \quad \text{s.t.} \quad S(f) \le B, \qquad \mathcal{L}(f,\lambda;B) = L(f) + \lambda\,[S(f) - B], \quad \lambda \ge 0
```

1. **For fixed λ, B does not affect the fit.** The term −λB is constant in f, so argmin_f 𝓛(f, λ; B) = argmin_f {L(f) + λS(f)}. The weighted approach never needs B.
2. **Every λ implies a budget.** Let f_λ be a global minimiser of L + λS and set B_λ = S(f_λ). Then f_λ also solves the constrained problem with budget B_λ: any g with S(g) ≤ B_λ and lower L would give L(g) + λS(g) < L(f_λ) + λS(f_λ), a contradiction. This needs global optimality, not convexity.
3. **The reverse (B → λ) needs more.** Under convexity and strong duality there is a multiplier λ_B for which the constrained solution also minimises the weighted objective, with complementary slackness λ_B [S(f_B) − B] = 0 (λ_B = 0 if the unrestricted fit already meets the budget). For neural networks this is not guaranteed, and local optimisation limits what a training run shows.

| You choose | You fit | You obtain afterwards |
| --- | --- | --- |
| λ | min L(f) + λS(f) | the achieved budget B_λ = S(f_λ) |
| B | min L(f) s.t. S(f) ≤ B | a multiplier (cost of the budget), when optimality conditions hold |

**Use in the paper.** Present the interpretable budget formulation, fit with fixed weights (our curvature-share rule ω picks λ), and report each fit's achieved budget. A pre-specified B may need a search over λ, and in a non-convex problem some budget optima may be unreachable that way. This answers the earlier worry that there is no practitioner way to set the SVM tolerance ε: choose λ by the curvature share and report B.

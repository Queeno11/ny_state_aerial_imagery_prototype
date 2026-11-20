"""Write descriptive comparisons from the completed persistent-error pilot."""
from pathlib import Path
import argparse
import json
import pandas as pd
from prior import HERE, SIMPLE

LABELS={
    "Ordinal_S":"Ordinal + stability",
    "Ordinal_L2_S":"Ordinal + levels + stability",
    "Ordinal_L2_D_S":"Full loss + stability",
    "Gaussian_ML_S_working_independence":"Gaussian ML + stability, independence working model",
    "Gaussian_ML_S_correlated_oracle":"Gaussian ML + stability, oracle pair covariance",
}
CONTROL={
    "Ordinal_L2_S":"No W-change loss",
    "Ordinal_L2_D_all_matched":"All tracts",
    "Ordinal_L2_D_changed_matched":"Changed tracts only",
    "Ordinal_L2_D_all_massmatched":"All tracts, lower weight",
}


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |",
                       "| "+" | ".join(["---"]*len(headers))+" |",
                       *["| "+" | ".join(map(str,row))+" |" for row in rows]])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    run=args.run
    df=pd.read_csv(run/"five_rep_summary.csv")
    old=pd.read_csv(SIMPLE/"calibration_10pct/five_rep_summary.csv")
    paired=pd.read_csv(run/"paired_summary.csv")
    weights=json.loads((run/"frozen_selection.json").read_text())["weights"]
    audit=json.loads((run/"audit.json").read_text())
    opt=pd.read_csv(run/"optimizer.csv")
    detection=pd.read_csv(run/"figures/detection_power_summary.csv")
    def cell(data,case,method,target,panel="heldout_unseen"):
        row=data[(data.case==case)&(data.method==method)&(data.target==target)&(data.panel==panel)]
        assert len(row)==1
        return row.iloc[0]
    text=["# Persistent tract errors: five matched replications", "",
          "The new residual law keeps the old marginal outcome variance and splits it equally between a tract component and transitory noise. The tract component persists only if the tract is unchanged. The ordinal baseline's mean structural-change MSE fell relative to the old independent-noise run, but its structural-level MSE rose. After retuning, the cubic level-loss branch improved structural levels substantially while worsening small-change detection. The controlled comparisons still show no mean structural-change gain from adding the difference loss at the tested coefficients.", "",
          "## New error structure", "",
          "For a city with total residual SD σ_c, draw v, v′, u₀ and u₁ independently with mean zero and variance σ_c²/2. Set e₀ = v + u₀ and e₁ = (1−D)v + Dv′ + u₁, where D is the simulator's indicator of genuine structural change. The outcome remains W = ψ_c(f(X)) + e. The distribution σ_c ∼ Uniform(.2,.6), structural features, acquisition features, city maps, true changes and original tract splits are unchanged.", "",
          table(["Quantity","Unchanged tract","Changed tract"],[["Endpoint residual variance","σ_c²","σ_c²"],["Cross-date residual correlation",".5","0"],["Difference residual variance","σ_c²","2σ_c²"]]), "",
          "Thus this design does not double the old noise. Difference noise is halved on stable tracts and unchanged on changing tracts. The primary outcome target is the conditional mean ψ_c(f), integrating over both residual components. The innovation in a genuinely unobserved amenity remains unpredictable from X; these figures do not claim to recover its realized individual change.", "",
          "## Estimation and calibration", "",
          "Five replications reuse the original structural draws and city effects. Each original city has 200 paired tracts: 160 fitting and 40 validation. All methods use the same exact acquisition-stability links. Primary loss coefficients were retuned on the independent replication 101 using the original 39-candidate protocol and observed validation objective. No evaluation data selected weights. Two additional branches retain the old ordinal-level and full-loss coefficients, isolating their numerical weights from retuning. Variance normalizers are recomputed from each fitting sample.", "",
          "All scores are frozen for the common cardinal comparison. Every method refits its map from scratch by identical unpenalized level least squares on 28 paired calibration tracts per city and is evaluated on 252 disjoint paired tracts. The 20 seen and 20 unseen cities, features and calibration/test splits match the previous 10% experiment. New residual innovations use a separate namespace. The outcome heads in A/B/C are the same families as before: A city intercept plus common positive slope; B city positive affine; C city monotone cubic. Maps remain fixed over dates.", "",
          "The original Gaussian and mixed-model benchmarks ignore the new residual correlation and are working models. Two additional Gaussian benchmarks know the true changed/unchanged status and covariance shape (correlation .5 or zero), but estimate the city SDs and all mean parameters. They are explicitly oracle references. Their GLS head and profiled variance equations reduce to the old likelihood at zero correlation. All methods receive the same level-only calibration afterward, including the oracle-likelihood scores.", "",
          "No hard-invariance estimator, noisy-stable-pair experiment, or additional neural architecture was implemented in this battery.", "",
          "## Frozen weights", ""]
    rows=[]
    for c in "ABC":
        w=weights[c]
        rows.append([c,f"{w['level']:.6g}",f"{w['full'][0]:.6g}",f"{w['full'][1]:.6g}",w['grid_winner']])
    text += [table(["Map","Level-only branch: level weight","Full branch: level weight","Full branch: change weight","Overall validation-grid winner"],rows),"",
             "Rank weight is 1 and stability weight .3. Each matched all/changed/lower-weight branch uses the level-only branch's level coefficient and the full branch's change coefficient. Changed-only averages the masked squared residual difference over all pairs; it does not double each retained pair's weight. The lower-weight all-tract control multiplies the change coefficient by the actual fitting changed fraction. Masks use true simulation status. Curvature variants are retained in the full data.","",
             "The independently tuned full and level branches are packages with different level weights. In C the full branch's level coefficient is one-tenth of the level-only branch's coefficient. Its better change performance cannot therefore be attributed to adding the difference term. The matched comparisons below isolate that term.","",
             "## Four comparable tables", "",
             "Each cell is mean city MSE ± city SD. We first calculate the equal-city mean and sample SD across 20 cities within each replication, then average those statistics over five replications. The SD describes city dispersion, not a standard error. The machine-readable summary also retains per-replication ranges. These are five-shot descriptive results.",""]
    for target,title,scale in [("f","Structural levels f",1e6),("W_conditional_mean","Conditional-mean outcome W",1e3),
                               ("delta_f","Structural changes Δf",1e6),("delta_W_conditional_mean","Conditional-mean outcome changes ΔW",1e6)]:
        rows=[]
        for case in "ABC":
            for method,label in LABELS.items():
                entries=[]
                for panel in ("heldout_seen","heldout_unseen"):
                    r=cell(df,case,method,target,panel)
                    entries.append(f"{r.mean_city_mse*scale:.4g} ± {r.mean_within_rep_city_sd*scale:.4g}")
                rows.append([case,label,*entries])
        text += [f"### {title}","",f"Values in units of {'10⁻⁶' if scale==1e6 else '10⁻³'}.","",
                 table(["Map","Scorer","Held-out seen cities","New cities after calibration"],rows),""]
    text += ["[Full mean/SD table](run_5rep/five_rep_summary.csv) · [All city rows](run_5rep/city_metrics.csv) · [Replication summaries](run_5rep/per_rep_city_summary.csv)","",
             "## Same old weights, different error law", "",
             "The following comparison uses the ordinal baseline, which already has identical coefficients in both designs. Features and city effects are matched; residual draws change. MSE units are 10⁻⁶.",""]
    rows=[]
    for c in "ABC":
        rows.append([c,*[f"{cell(d,c,'Ordinal_S',t).mean_city_mse*1e6:.3f}" for t in ("f","delta_f") for d in (old,df)]])
    text += [table(["Map","Old f","Persistent f","Old Δf","Persistent Δf"],rows),"",
             "The complete same-old-coefficient comparison also includes ordinal + levels and full loss. It uses unseen cities and the same cardinal calibration protocol. This isolates the specified loss coefficients; it does not hold estimated normalizers or fitted representations fixed. Five replications do not establish a universal effect of persistence.","",
             "[Same-weight figure](run_5rep/figures/fixed_weight_noise_comparison/equal_calibration_comparison.png) · [Comparison data](run_5rep/figures/fixed_weight_comparison_data.csv)","",
             "## Matched difference-loss experiment", "",
             "Unseen structural-change MSE, in units of 10⁻⁶:",""]
    rows=[]
    for c in "ABC":
        rows.append([c,*[f"{cell(df,c,m,'delta_f').mean_city_mse*1e6:.3f}" for m in CONTROL]])
    text += [table(["Map",*CONTROL.values()],rows),"",
             "Adding the difference term raises mean structural-change MSE in each map among these matched arms. Changed-only is also higher than the lower-weight control in all three means. The effects in B/C are small at their selected coefficients. These findings concern the tested weights, not the existence of an optimal positive difference coefficient.","",
             "[Matched mean-error figure](run_5rep/figures/matched_comparison/equal_calibration_comparison.png) · [Matched city SD](run_5rep/figures/matched_comparison/city_variability.png) · [Paired differences, ranges and counts](run_5rep/paired_summary.csv)","",
             "## Small-change detection", "",
             "Each score gets a cutoff from a separate independent stable panel, using the empirical 95th percentile of absolute score movement. Fine change doses were evaluated without retraining or tuning. The numbers below average power over positive/negative changes and five replications on unseen cities. Doses .002–.05 expose differences that were hidden by near-perfect detection at the older larger doses.",""]
    rows=[]
    for c in "ABC":
        for method,label in LABELS.items():
            values=[]
            for dose in (.005,.01,.02):
                r=detection[(detection.case==c)&(detection.method==method)&(detection["where"]=="unseen") & (detection.magnitude==dose)]
                assert len(r)==1
                values.append(f"{r.iloc[0]['mean']*100:.1f}%")
            rows.append([c,label,*values])
    text += [table(["Map","Scorer","Change .005","Change .01","Change .02"],rows),"",
             "Fresh stable false-alarm rates are approximately 4.71–4.72%. The same linear acquisition-noise scale cancellation discussed in the original pilot applies. At change .005, ordinal power is about 84–85% in all three unseen-city cases. In C, the retuned level-only branch improves mean f-level MSE by about 45% relative to ordinal but lowers power at .005 to about 50%. The full C branch has a smaller level weight and about 57% power; this is not evidence that the difference term caused that gain. The oracle covariance likelihood does not dominate detection in these five runs; all likelihood comparisons with stability also depend on the fixed numerical stability weight.","",
             "[Detection figure](run_5rep/figures/small_change_detection.png) · [Detection data](run_5rep/detection.csv) · [Fresh false alarms](run_5rep/figures/detection_false_alarms.csv)","",
             "## Audit and reproduction", "",
             f"Completed {audit['scorer_fits']} scorer fits plus 39 independent tuning candidates and 560 common calibration head sets. All 560 calibration checks preserved scores exactly. Largest calibration gradient was {audit['max_calibration_gradient']:.3g}; largest selected score-gradient norm was {opt.min_tangent_grad.max():.3g}. Masked-loss finite differences and correlated-likelihood finite differences passed. A separate 500,000-pair noise diagnostic reproduced endpoint variance 1, stable covariance .5 and difference variance 1, and changed covariance 0 and difference variance 2 after normalizing by city sigma.","",
             "The two-stage index mixed model converged in 15/15 fits. The diagonal sensitivity model converged in 9/15, with warnings retained; neither models the new tract correlation. Original model files remain unchanged. The source manifest records both new and reused implementations. The stronger invariance proposal remains a proposed estimator, not a result of this run.","",
             "[Audit](run_5rep/audit.json) · [Noise and likelihood preflight](run_5rep/preflight.json) · [Manifest](run_5rep/manifest.json) · [Optimizer diagnostics](run_5rep/optimizer.csv)","",
             "Run from this folder, choosing a fresh run directory:","","```text",
             "python run_persistent.py --out NEW_RUN","python plot_persistent.py --run NEW_RUN",
             "python report_persistent.py --run NEW_RUN","```", ""]
    destination=HERE/"ANALYSIS.md"
    destination.write_text("\n".join(text),encoding="utf-8")
    print(destination)


if __name__=="__main__":
    main()

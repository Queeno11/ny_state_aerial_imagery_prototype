"""Figures for the persistent-error pilot, including a fixed-weight comparison."""
from pathlib import Path
import argparse

from prior import HERE, SIMPLE
import plot_calibration_10pct as plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PRIMARY = [
    ("Ordinal_S", "Ordinal", "#2166AC"),
    ("Ordinal_L2_S", "Ordinal + levels", "#15877B"),
    ("Ordinal_L2_D_S", "Full loss", "#DB6537"),
    ("Gaussian_ML_S_working_independence", "ML: independent errors", "#8056A1"),
    ("Gaussian_ML_S_correlated_oracle", "ML: oracle pair covariance", "#243444"),
]
CONTROLLED = [
    ("Ordinal_L2_S", "No W-change loss", "#15877B"),
    ("Ordinal_L2_D_all_matched", "All tracts", "#B83C52"),
    ("Ordinal_L2_D_changed_matched", "Changed tracts only", "#A18316"),
    ("Ordinal_L2_D_all_massmatched", "All tracts: lower weight", "#8056A1"),
]
NOISE_NOTE = "Total residual variance unchanged. Half tract component + half transitory; tract component is redrawn after change."


def draw_set(data, methods, destination, mode="persistent", **kwargs):
    plot.DATA, plot.METHODS, plot.OUT = data, methods, destination
    destination.mkdir(parents=True, exist_ok=True)
    plot.draw("mean_city_mse", mode=mode, **kwargs)
    plot.draw("mean_within_rep_city_sd", dispersion=True, mode=mode, **kwargs)


def detection_figure(run, out):
    raw = pd.read_csv(run / "detection.csv")
    raw = raw[raw.method.isin([m[0] for m in PRIMARY])].copy()
    raw["magnitude"] = raw.dose.abs().round(6)
    positives = raw[(raw.magnitude > 0) & (raw.magnitude <= .050001)]
    per_rep = positives.groupby(["rep", "case", "method", "where", "magnitude"]).flag_rate.mean()
    summary = per_rep.groupby(["case", "method", "where", "magnitude"]).agg(["mean", "min", "max"]).reset_index()
    summary.to_csv(out / "detection_power_summary.csv", index=False)
    false_alarms = raw[raw.dose == 0].groupby(["case", "method", "where"]).flag_rate.agg(["mean", "min", "max"])
    false_alarms.to_csv(out / "detection_false_alarms.csv")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8), facecolor="#FAFBFD")
    fig.subplots_adjust(left=.075, right=.98, top=.74, bottom=.15, wspace=.14, hspace=.25)
    fig.text(.055,.95,"PERSISTENT TRACT ERRORS  /  FIVE-REPLICATION PILOT",fontsize=10,weight="bold",color="#60788B")
    fig.text(.055,.9,"Can the score detect smaller changes?",fontsize=25,weight="bold",color="#182D40")
    fig.text(.055,.858,"Each method uses a separate stable-sample cutoff targeting a 5% false-alarm rate.",fontsize=11.5,color="#526171")
    handles=[Line2D([],[],color=c,marker="o",label=l,lw=2) for _,l,c in PRIMARY]
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.05,.827),ncol=3,frameon=False,fontsize=10)
    for row,where in enumerate(("seen","unseen")):
        for col,case in enumerate("ABC"):
            ax=axes[row,col]
            ax.set_facecolor("white")
            for method,label,color in PRIMARY:
                s=summary[(summary.case==case)&(summary.method==method)&(summary["where"]==where)].sort_values("magnitude")
                ax.plot(s.magnitude,s["mean"]*100,color=color,marker="o",ms=4,lw=1.8)
            ax.axhline(80,color="#9AABB8",ls="--",lw=.8)
            ax.set_xscale("log")
            ax.set_xticks([.002,.005,.01,.02,.05],[".002",".005",".01",".02",".05"])
            ax.set_ylim(0,103)
            ax.set_yticks([0,20,40,60,80,100])
            if col==0:
                ax.set_ylabel(("Training cities" if where=="seen" else "New cities")+"\nDetection probability (%)")
            else:
                ax.set_yticklabels([])
            if row==0:
                ax.set_title({"A":"A  |  Intercept","B":"B  |  Affine","C":"C  |  Cubic"}[case],loc="left",weight="bold")
            else:
                ax.set_xlabel("Absolute true change in f  ·  log scale")
            ax.grid(axis="y",color="#E0E6EC",lw=.7)
            ax.spines[["top","right","left"]].set_visible(False)
            ax.tick_params(axis="both",which="both",length=0,pad=6)
    fpr_low,fpr_high=false_alarms["mean"].min()*100,false_alarms["mean"].max()*100
    fig.text(.055,.075,f"Fresh stable false-alarm rates: {fpr_low:.2f}%–{fpr_high:.2f}%. Higher power is better at comparable false-alarm rates.",fontsize=11,weight="bold")
    fig.text(.055,.045,"Curves average positive and negative changes, then five replications. Dashed line marks 80% detection.",fontsize=9.5,color="#607181")
    fig.text(.055,.022,"All methods use stable acquisition pairs. Oracle likelihood knows which training tracts changed and their covariance shape.",fontsize=9,color="#607181")
    for ext in ("png","pdf","svg"):
        fig.savefig(out/f"small_change_detection.{ext}",dpi=190)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    data=pd.read_csv(args.run/"five_rep_summary.csv")
    root=args.run/"figures"
    plot.PANELS=[("heldout_seen",True),("heldout_unseen",False)]
    draw_set(data,PRIMARY,root/"main",study_label="PERSISTENT TRACT ERRORS  /  FIVE-REPLICATION PILOT",
             figure_subtitle="Same 10% local calibration. Primary loss weights are retuned on a separate calibration replication.",
             footer_headline="Oracle likelihood knows the pair-correlation pattern; working-independence likelihood ignores it.",
             extra_note=NOISE_NOTE)
    draw_set(data,CONTROLLED,root/"matched_comparison",mode="controlled",
             study_label="PERSISTENT TRACT ERRORS  /  MATCHED LOSS WEIGHTS",
             footer_headline="Changed-only uses true change status; the lower-weight control matches its total difference-loss weight.",
             extra_note=NOISE_NOTE)
    old=pd.read_csv(SIMPLE/"calibration_10pct/five_rep_summary.csv")
    keys={"Ordinal_S":"Ordinal_S","Ordinal_L2_S_oldweights":"Ordinal_L2_S","Ordinal_L2_D_S_oldweights":"Ordinal_L2_D_S"}
    old=old[(old.panel=="heldout_unseen") & old.method.isin(keys.values())].copy()
    old["panel"]="independent"
    new=data[(data.panel=="heldout_unseen") & data.method.isin(keys)].copy()
    new["method"]=new.method.map(keys)
    new["panel"]="persistent"
    fixed=pd.concat([old,new],ignore_index=True)
    plot.PANELS=[("independent",True),("persistent",False)]
    draw_set(fixed,PRIMARY[:3],root/"fixed_weight_noise_comparison",
             study_label="ERROR-STRUCTURE COMPARISON  /  OLD LOSS COEFFICIENTS HELD FIXED",
             figure_title="What changes when tract errors persist?",
             figure_subtitle="New-city evaluation; same structural draws, city maps and calibration splits. Models are refitted.",
             panel_labels=("Original independent errors","Half persistent + half transitory"),
             footer_headline="Each endpoint has the same residual variance in both designs. Loss coefficients are held fixed.",
             extra_note="Variance normalizers are recomputed from each fitting sample. W targets exclude both residual components.")
    fixed.to_csv(root/"fixed_weight_comparison_data.csv",index=False)
    detection_figure(args.run,root)
    print(root)


if __name__=="__main__":
    main()

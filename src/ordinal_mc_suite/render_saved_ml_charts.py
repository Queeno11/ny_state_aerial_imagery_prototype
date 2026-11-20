"""Render augmented saved-run charts, including compact overviews for sharing."""
from pathlib import Path
import zipfile
import matplotlib.pyplot as plt
import presentation_figures as pf

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'deck_figures_with_ml'; pf.setup_style()
    pf.build_all(out); pf.build_e1_additive(ROOT/'e1_additive_exact',out)
    e1=pf.load_csv(ROOT/'overnight_measurement','summary.csv')
    fig,axes=plt.subplots(2,2,figsize=(8.8,5.7),sharey=True)
    for row,composition in enumerate((0.,.7)):
        for col,(stage,_) in enumerate(pf.E1_PANELS):
            ax=axes[row,col]
            rows=pf.select(e1,metric='unseen_within_latent',composition=composition,psi_stage=stage)
            curves=[pf.curve(pf.select(rows,method=m),label,color,linestyle=ls)
                    for m,label,color,ls in pf.E1_METHODS]
            pf.draw_curves(ax,pf.with_ml(curves,rows))
            ax.set_title(f'Composition = {composition:g}; '+('multiplier map' if stage=='multiplier' else 'level + shape map'),loc='left')
            ax.set_xlabel('labeled training cities $C$')
        axes[row,0].set_ylabel('misranked pairs, unseen cities (%)')
    pf.finish_legend(fig,axes[-1,-1])
    pf.save(fig,out/'mc_e1_overview.pdf')
    nn=pf.load_csv(ROOT/'overnight_nn','summary.csv')
    linear=pf.load_csv(ROOT/'linear_extended','summary.csv')
    fig=pf.fig_links([(linear,{},'linear truth / linear fit (50 reps)'),
                      (nn,{'truth':'linear'},'linear truth / NN fit (30 reps)'),
                      (nn,{'truth':'quadratic'},'quadratic truth / NN fit (30 reps)')])
    pf.save(fig,out/'mc_e3_overview.pdf')
    # Put every regenerated plot, plus the finished rho chart, in one download.
    with zipfile.ZipFile(out/'charts_with_ml.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.iterdir()):
            if p.suffix in ('.pdf','.png'): archive.write(p,p.name)
        for p in sorted(ROOT.parent.glob('rho_*/experiment_summary_with_ml.png')):
            archive.write(p,p.parent.name+'/'+p.name)
    print(out,flush=True)


if __name__=='__main__': main()

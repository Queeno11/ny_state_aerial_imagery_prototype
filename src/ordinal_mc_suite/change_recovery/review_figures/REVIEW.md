# E5 review

All 5,760 selected fits are present, with 30 replications in every cell
and no duplicate design/replication rows. Independently recalculated means,
Monte Carlo standard errors, and confidence intervals match 960
saved summary cells. Every ML fit reports convergence.

The key qualification is optimization: all 1,440 linked
NN fits fail the objective audit. Their additional audit optimization lowers the
objective materially (see optimization_audit.csv). Unlinked NN fits and all linear
fits pass that audit. These results reveal a simulated NN recovery problem but do
not distinguish inadequate optimization from statistical/representation limits.
They therefore do not establish that the empirical difficulty is application-only.

At feature step 0.4 with no contaminated training links, NN detection recall is
about 15.1% (linear truth) and 16.4% (quadratic truth) with temporal links, versus
6.3% and 6.5% without links. Temporal + cross-city recall is 13.9% and 14.2%.
The correctly specified linear-fit control detects essentially all changes under
linear truth. ML detects essentially all changes for both truths. Linear fits
under quadratic truth detect changes well with links but have a change slope
near 0.5, illustrating why detection alone is insufficient.

False-positive rates for the ranking estimators are around the intended 5%.
Contaminating training links further reduces NN recall. The plotted NN performance
is provisional because of the audit failures. No fits were rerun or modified.

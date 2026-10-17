Make a first draft of a 40' presentation of my research agenda for the prof jaume Vives-i-Bastida Lab (so details are relevant, and he will ask them). I have the following general structure I think is good, but happy to have suggestions. You can use the attached documents, several contradict each other because some are newer. The truth now is what's here in the bullets, but some motivations/ideas can come from those docs):  


Intro
- Motivation: why satellite imagery? Make this point strong with some examples on how it is being used to target programs (if anywhere) or how there is demand in UN/WB documents for this
- The big (and far) goal: what about time series? can we measure this over time? At least for now... not in a useful way. But picture the potential: (a) train a "global" model (b) finetune with 1 or 2 national surveys (in most countries you have some of this data) (c) use this to generate high resolution panel dataset of predictions of wealth (d) use these predictions to make policy evaluations where you dont have temporal data (DiD, RDD, whatever) => potential to scale impact evaluation globally at a low cost
- An ambitius (maybe too ambitious) research agenda: maybe 4 papers, maybe 1 paper, we'll se on the way! will present my current state in a) and b)
    a) The measurement problem: what are the issues to measure this + some solutions that generalize to other setups (~30% done)
    b) Predicting wealth per capita in the US: prove that this indeed works for the US and outperforms other methods (~60% done)
    c) How to create proper time series (0% done)
    d) Show the potential of these predictions into several examples (0% done)
- What about the literature? Isn't is there already? Yes and no:
    a) you have excellent AI models detecting buildings, predicting land usage, detecting damages from natural disasters, etc
    b) good models that predict DHS Wealth index => Jean et al, Newhouse papers, latest newhouse paper on temporal predictions on DHS Wealth for africa, Blumenstock, others
    c) models that predict income/wealth directly and perform ok but not that good => Khachiyan et al, Abbate et al, others
    d) None has achieved time-series/stable generation of monetary measures,

1) The measurement problem: Learning a Common Index from Outcomes Measured on Non-Comparable Scales
(working on this for Honors Machine Learning final project)

- The puzzle: why can visual models scale across geography for buildings but not for wealth? 
    - show 6 pics of aerial imagery: 2 rich from NYC and chicago that look similar, 2 medium class from NYC/Chicago that look similar and 2 poor from NYC/Chicago that look alike
- The puzzle (2): overlay building footprint for those imges
- The puzzle (3): put under the wealth they have, to highlight that they represent different things
- So the model needs to be able to separate real features from the images from incidental changes (show NYC images "E:\Datasets\Imagenes Satelitales\New York City\nyc_2010.png" to "E:\Datasets\Imagenes Satelitales\New York City\nyc_2024.png") to make the point of "incidental changes"

- The model: First show the general W = gamma(X)+u  equation, text each component: X is observable characteristics (either a vector of covariates, images, both, whatever) and gamma and u,
- The model (2): show eq but say that (I assume) gamma can be decomposed between a cell (city-year) component (different cost of living and wages, inflation, etc) and a "global" component that says that some features make some tracts richer than others (example using the images in prev slide). We could estimate gamma_ct directly but requires assuming either linear FE or fitting a specific function per cell, which break out of sample validity. The goal is to estimate f without caring much about psi. Make a clear bracket saying that we care about f(), that is our estimand
- Assuming monotonicity (which says in some way that the only thing that gamma is doing is rescaling the features to city-year-level prices) in gamma lets us express W as W=psi(f(x)+e) and define R := f(X) + e. Here R represents a latent wealth index, only determined by the features in X and the errors, and  after that is transformed via psi to get the wealth. R formalizes the intuition from the images before, even if the numbers are different, there is a clear ordinal ranking of Aricher than B richer than C
- State briefly Proposition 1

How can we learn about R?
- State Assumption 1-3 (make each assumption appear 1-by-1 so they are discussed in time): monotonicity, invariance and regularity (lipz) (use the notation from the bottom of this txt) (under the lipz assumption state that the L will have a different interpretation and magnitud in each setup: in sat img, stable pairs can be an independent model that detects changes in images and say no change, a no building footprint change between the two moments, etc) 
- State A4 (three versions, say I only did the strongest version but plan to solve the others)
- What can we learn about R? Proposition 2 (no proof, but explain in 1-line)
- State the criterion with its two components (Definition 2)
- State Theorem 1 under A4(iii) (for (i) put it as Q(rhof)-Q(s)<=lambda L2 delta2, its more intuitive)
Estimation
- State A5 and A6 and convergence Theorem 2 of shat to rho f. Explain intuitively the bands: its Theorem 1(ii) with alpha_theta, which is the approximation error of the class.

Ok good it converges but why use this? => Montecarlo simulations
- General setup parameters
Slide	Question	What to show	Intended takeaway
1. Non-comparable labels	Why estimate using rankings?	Experiment 1: pooled L2, a feasible parametric correction, and unlinked common logit. Two panels: correction correctly specified versus misspecified.	Ranking discards cardinal information but protects against unknown monotone measurement transformations.
2. Additional information from stability	Why add temporal pairs?	Your completed experiment 3: linear truth and linear estimator, no links versus temporal links, 8–64 cities.	Valid stability restrictions improve prediction even when the ranking model is correctly specified.
3. Flexible prediction	Does the gain survive a flexible estimator?	Regularized NN experiment 3, no links versus temporal links. Two panels: linear truth and one prespecified nonlinear truth, such as quadratic.	Test whether temporal information helps beyond ordinary regularization.
Optional 4. Imperfect stability	What if the restrictions are wrong?	Experiment 4: exact links, modest drift, and false links in three panels, with identical axes and methods.	Establish the cost of relying on approximate or incorrect restrictions.


2) Predicting wealth per capita in the US: Zooming in from Outer Space: A Building-Level AI Framework for Tracking Wealth

- We are gonna use the ecoometric framework (using logistic errors assumption) to measure wealth in the US: state the data used: images from NAIP, data at census tract level from 5-year ACS
- W is Wealth: state the definition of wealth here
- X are the images, put 3 examples of images
- State the loss again, show the batch Composition image (both from the beamer slides) (I will probably update the figures tho) and the train-test-val splits

Results
- (same as beamer) Cross sectional oos performance
- ordinal beats L2 systematically: only in 1 city in the test set it improve wrt ordinal estimation. ("results\run_20260722\evaluation\figures\US_L2_city_gap_vs_signed_distance.pdf")
- (same as beamer) different cities as examples
- some viz of autocorrelation? the table is ok but boring...
- (same as beamer) CSA analysis. Now with the framework, state that if ||h-h'|| is larger (because h and h' are not stable pairs) then |g(h)-g(h')| might increase (might not too if the new buildings are similar to the old ones! NYC is great because most new construction is changing very old buildings with new high end ones, so the g(h) should move!!) => testing in NYC aerial imagery + NYC buildings footprints: != imagery source on tracts never seen. (recall to update the notation to our new notation!)
- (same as beamer) show NYC result with anticipation 2 ("results\run_20260722\nyc_zarr_check\evaluation\csa_anticipation2\figures\D_event_study_thresholds_nyc.pdf") (robust to other anticipation)
- Add a nice full slide NYC buildings prediction per deciles
- The challenge: estimating delta f:  ("ny_state_aerial_imagery_prototype\results\run_20260722\nyc_zarr_check\evaluation\figures\F_construction_scatter.pdf")


Conclusions:
- (partially) Built an econometrics framework for this
- MC shows it improves performance even in linear models
- US application that shows it works
- The model seems to behave as expected, even tho the time trayectories are not
- Next steps:
    Improve theoretical framework
    - Remanining convergence propositions
    - Some small theorems that explain clearly why this works? My idea is that it reduces the class of functions where optimization can ocurr, so it trains faster (and with smaller n!)
    - Can we put some assumptions on the errors to learn temporal change properly?

    Improve model training:
    - better hyperparameter tuning (training was one-shot of my lambda hyperparameter)
    - harder augmentation to make NYC aerial perform better (maybe a few fine tuning epochs?)
    - Try other assumptions on the errors, might improve performance!

    What about inference? add skull emoji
    




Next steps: 


Put "results\run_20260722\evaluation\figures\US_L2_training_curves.pdf" into the appendix








Units are indexed by $i$, groups by $c$ (cities) and dates by $t$; the pair $(c,t)$ is a
\emph{cell}. The data are a label $W_{ict}\in\R$ and covariates $X_{ict}$, an image tile in the
application. The model is
\begin{equation}\label{eq:model}
  W_{ict} \;=\; \xi_{ct}(X_{ict}) + e_{ict},
  \qquad
  \xi_{ct} \;=\; \psi_{ct}\circ f .
\end{equation}
The \emph{global function} $f$ maps covariates into a latent index and is the same in every
cell. The \emph{cell-specific function} $\psi_{ct}$ converts the index into the units in which
the cell reports it (dollars in city $c$ in year $t$) and carries that cell's level, spread and
shape. A model fitted cell by cell learns $\xi_{ct}$, which says nothing about a cell without
labels. The goal is to estimate $f$ without estimating $\psi_{ct}$.
 
Let $\Sset$ be a set of \emph{stable pairs} $(x,x')$: covariate values observed in two
different cells and selected, without reference to the labels, as nearly the same unit.
 
\begin{assumption}[Monotonicity]\label{a:mono}
$\psi_{ct}$ is strictly increasing for every $(c,t)$.
\end{assumption}
\begin{assumption}[Invariance]\label{a:inv}
$f$ does not depend on $c$ or $t$.
\end{assumption}
\begin{assumption}[Regularity]\label{a:lip}
There is a factorization $f=g\circ\phi$, with the coordinates of $H=\phi(X)\in\R^K$
standardized, such that $|g(h)-g(h')|\le L\,\|h-h'\|$ on the support of $H$, and every stable
pair $(x,x')\in\Sset$ satisfies $\|\phi(x')-\phi(x)\|\le\delta$. The index is not constant
within cells.
\end{assumption}
 
\ref{a:inv} is the substantive assumption: all variation across cells is assigned to
$\psi_{ct}$. In \ref{a:lip}, $H$ is the embedding before the final scalar layer of a network,
and $L$ and $\delta$ matter only through their product.
 
By \ref{a:mono}, $\psi_{ct}$ is invertible, so the error in \eqref{eq:model} can be moved inside
$\psi_{ct}$ after rescaling it. Taking the range of $\psi_{ct}$ to contain the labels, define the
latent index $R_{ict}=\psi_{ct}^{-1}(W_{ict})$ and $\varepsilon_{ict}=R_{ict}-f(X_{ict})$, which
is $e_{ict}$ expressed in units of the index. Then \eqref{eq:model} becomes
\begin{equation}\label{eq:latent}
  W_{ict} \;=\; \psi_{ct}(R_{ict}),
  \qquad
  R_{ict} \;=\; f(X_{ict}) + \varepsilon_{ict} .
\end{equation}
The equation for $R$ reads like a linear model: a common function of the covariates plus an
additive error. The difficulty is that $R$ is never observed, only its cell-specific
transformation $W$.
 
\begin{proposition}[Pooled levels estimate a mixture]\label{p:mix}
Suppose \eqref{eq:model} holds with $\E[e_{ict}\mid X_{ict},c,t]=0$ and $\E|W|<\infty$, and $W$ is
regressed on $X$ pooling all cells. Then
\[
  \E[W\mid X=x] \;=\; \sum_{c,t}\Pr(c,t\mid X=x)\;\psi_{ct}\bigl(f(x)\bigr).
\]
\end{proposition}
 
The weights reflect which cells are in the sample, and they can reverse the ranking. If public
housing towers appear only in New York and single-family houses mostly in Dallas, a large enough
dollar gap between the cities makes the pooled fit rank the towers above the houses, although
they are poorer in each city: the model has learned that public housing signals New York.
 
\begin{proposition}[What the data reveal about the index]\label{p:reveal}
Let \ref{a:mono}--\ref{a:lip} hold.
\begin{enumerate}[label=\textup{(\roman*)}]
\item \textup{Ordering within a cell.} For two units $i,j$ of one cell $(c,t)$:\;
      $W_{jct}>W_{ict}\iff R_{jct}>R_{ict}$.
\item \textup{Stability across cells.} For every stable pair $(x,x')\in\Sset$, with $x$ observed
      in cell $(c,t)$ and $x'$ in cell $(c',t')\neq(c,t)$:\; $|f(x')-f(x)|\le L\delta$.
\end{enumerate}
\end{proposition}
\begin{proof}
(i) $W=\psi_{ct}(R)$ for both units, with the same strictly increasing $\psi_{ct}$
(\ref{a:mono}). (ii)~By \ref{a:inv} the same $f=g\circ\phi$ applies in both cells, and by
\ref{a:lip} $|f(x')-f(x)|\le L\,\|\phi(x')-\phi(x)\|\le L\delta$.
\end{proof}
 
Part (ii) holds for any two cells, so a stable pair may join two dates of one group
($c=c'$, $t\neq t'$) or two different groups ($c\neq c'$). In satellite and aerial imagery only
the first kind arises on its own: one location, two years, no recorded structural change.
Cross-city pairs require an already pretrained model, in whose representation near neighbours
can be defined before training.
 
Proposition~\ref{p:reveal} establishes what is observable: the ordering of $R$ within cells and
near-equalities of $f$ across cells, neither involving $\psi_{ct}$. But $R$ is $f$ plus an
error, so whether a comparison of two labels reveals the ordering of $f$ depends on the errors,
and the assumption made about them determines the loss. For two units $i\neq j$ of one cell
write $\Delta_{ij}s=s(X_j)-s(X_i)$ for a score function $s$,
$y_{ij}=\one\{W_j>W_i\}=\one\{\varepsilon_i-\varepsilon_j<\Delta_{ij}f\}$ and
$p_{ij}=\Pr(y_{ij}=1\mid X_i,X_j,c,t)$.
 
\begin{assumption}[Comparison errors, three versions]\label{a:err}
Given $(X_i,X_j,c,t)$, the difference $\varepsilon_i-\varepsilon_j$
\begin{enumerate}[label=\textup{(\alph*)}]
\item \textup{Median zero:} has median zero and a distribution function strictly increasing at
      zero, so that $p_{ij}>\tfrac12\iff\Delta_{ij}f>0$;
\item \textup{Heterogeneous logistic:} is logistic with scale $1/\rho_{ct}>0$, so that
      $p_{ij}=\Lambda(\rho_{ct}\,\Delta_{ij}f)$;
\item \textup{Homogeneous logistic:} satisfies \textup{(b)} with $\rho_{ct}=\rho$ in every cell.
\end{enumerate}
\end{assumption}
 
Here $\Lambda(z)=1/(1+e^{-z})$. Version (a) lets the noise depend on the features; (b) and (c)
fix its shape, with and without a reliability that differs by cell. All three lead to one
criterion,
\begin{equation}\label{eq:crit}
  Q_k(s,\eta) \;=\; Q^{\mathrm{cross}}_k(s,\eta)
  \;+\;\lambda_t\,\frac{S_t(s)}{V(s)}
  \;+\;\lambda_c\,\frac{S_c(s)}{V(s)},
  \qquad k\in\{\mathrm{a},\mathrm{b},\mathrm{c}\},
\end{equation}
where $S_t(s)$ and $S_c(s)$ are the mean squared score differences $\E[(s(x')-s(x))^2]$ over
temporal and cross-group stable pairs, and $V(s)=\tfrac12\E[(\Delta_{ij}s)^2]$ is the within-cell
dispersion of the score. Dividing by $V$ makes the penalties scale-free, so shrinking the
scores cannot satisfy them; a weight is zero when that kind of pair is unavailable. The
criterion uses the labels only through $y_{ij}$, so it is unchanged by any strictly increasing
relabeling within a cell: it never sees $\psi_{ct}$.
 
Only the first term depends on the error assumption. Under median zero an ordinal loss makes
sense: the probability of misordering a pair,
$Q^{\mathrm{cross}}_{\mathrm a}(s)=\Pr\bigl((2y_{ij}-1)\Delta_{ij}s<0\bigr)+\tfrac12\Pr(\Delta_{ij}s=0)$,
trained through a smooth surrogate. Under heterogeneous logistic errors it is a RankNet loss
parametrized at the city-year level,
$Q^{\mathrm{cross}}_{\mathrm b}(s,\eta)=\E[\ell(\rho_{ct}\Delta_{ij}s,\,y_{ij})]$ with
$\ell(z,y)=\log(1+e^{z})-yz$, $\eta=\{\rho_{ct}\}$ and the scale of $s$ fixed by $V(s)=1$. Under
homogeneous logistic errors it is the RankNet loss of \citet{burges2005},
$Q^{\mathrm{cross}}_{\mathrm c}(s)=\E[\ell(\Delta_{ij}s,\,y_{ij})]$, with the scale absorbed
into $s$.
 
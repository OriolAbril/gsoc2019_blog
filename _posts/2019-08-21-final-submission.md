---
layout: post
title:  "Work done during GSoC"
description: "Submission for GSoC's final evaluation. Summary
of all the work done during the coding period."
date:   2019-08-21 00:22:28 +0200

---

To see all the work done during the 2019 GSoC coding period, see this
[pull request (PR) list](https://github.com/arviz-devs/arviz/pulls?page=1&q=is%3Apr+author%3AOriolAbril+archived%3Afalse+label%3AGSOC&utf8=%E2%9C%93).
This document contains all my contributions during GSoC coding period divided by category. In addition, the most
important contributions in each topic are summarized. To read more details on the new or modified functions,
see [ArviZ docs](https://arviz-devs.github.io/arviz/index.html), moreover, a direct link to each new or modified function is
available in its changelog summary. These are the different categories into which PRs have
been divided:

1. [Information Criteria](#information-criteria)
1. [Convergence Assessment](#convergence-assessment)
1. [Model Checking](#model-checking)
1. [InferenceData scheme modifications](#inferencedata-scheme-modifications)
1. [Global parameters](#global-parameters)
1. [Other Pull Requests](#other-pull-requests)

---
### Information Criteria

Information criteria allow to evaluate the predictive accuracy of a model. In some cases, this quantity is so
relevant to the analysis that it can be used to compare
different models and choose the one with the better predictive accuracy. ArviZ has some functions designed to calculate
different information criteria (`az.loo` and `az.waic`) and to analyze its results (`az.compare`, `az.plot_compare`,
`az.plot_elpd` and `az.plot_khat`). During GSoC, these functions have been modified in order to extend them and ease
their interpretation:

* A new class `az.ELPDData` was created. It is designed to store and print in an informative manner information criteria
results. See the docs on [`az.loo`](https://arviz-devs.github.io/arviz/generated/arviz.loo.html#arviz.loo) and
[`az.waic`](https://arviz-devs.github.io/arviz/generated/arviz.waic.html#arviz.waic) for examples.
* The computation of information criteria was modified to use internally `az.wrap_xarray_ufunc`. This allows 3 key improvements.
The first is that there is no longer the need to convert to unlabeled data, therefore pointwise information criteria
results are labeled like the rest of the data in InferenceData objects which eases the identification of problematic
observations. The second is that working with multidimensional data is automatically handled by xarray and the shape of
the original object is kept. And the third is that this change allows easy parallelization of the calculations using
dask via xarray.
* This previous change also defines by convenience ArviZ's default shapes and dimension names. InferenceData objects
generally contain first the `chain` dimension followed by the `draw` dimension and then any number of dimensions depending
on the variable shape. In some cases, we want to combine all chains into a single array; in ArviZ this should be
done with xarray's `stack(sample=("chain", "draw"))`. This combines the `chain` and the `draw` dimension to a single
dimension called `sample` which is placed as the last dimension, leaving an object with first the variable shape and then
the `sample` shape.
* A new plotting function [`az.plot_elpd`](https://arviz-devs.github.io/arviz/generated/arviz.plot_elpd.html#arviz.plot_elpd)
was created in order to compare pointwise information criteria values
of several models. Some of its options include coloring based on label values and non-degenerate pairwise
plots to compare more than two models at once.
* Many new customization options were added to
[`az.plot_khat`](https://arviz-devs.github.io/arviz/generated/arviz.plot_khat.html#arviz.plot_khat).
Some examples are coloring based on labels, a
summary of the quality of khat values and showing the labels of each observation when hovering over.
* The context manager
[`az.interactive_backend`](https://arviz-devs.github.io/arviz/generated/arviz.interactive_backend.html#arviz.interactive_backend)
was created to allow temporal change of matplotlib backend between the
ususal inline backend used in jupyter notebooks, jupyter lab or spyder and any of the supported interactive backends.
* Tests of information criteria functions were extended. Now, the behaviour of these functions is also tested on
multidimensional objects (`chain`, `draw` and at least 2 more dimensions).
* Work on [sampling wrappers](https://github.com/arviz-devs/arviz/pull/771)
was started in order to allow ArviZ to refit the same model on different sets of data.
As ArviZ has no sampling capabilities and it is backend agnostic, this wrappers allow it to perform refits and to do them
using any backend available like PyStan or emcee. At the time of writing, working sampling wrappers for PyStan and emcee
have already been written. PyMC3 wrappers cannot work yet until this
[PyMC3 issue](https://github.com/pymc-devs/pymc3/issues/3007) is fixed. The PR implementing this is still
unmerged as it also depends on the changes on InferenceData scheme.
* In parallel to the sampling wrappers, an ArviZ port of the [brms](https://github.com/paul-buerkner/brms)
reloo function was also written. This function uses the sampling wrappers to calculate exact cross validation values
for problematic observations where numerical approximations do not work. After GSoC, we also plan on implementing a
numerical approximation of the leave-future-out cross validation (which also needs to perform some refits of the model).


#### PRs in this caterory
* [#678](https://github.com/arviz-devs/arviz/pull/678): +809 −102
* [#700](https://github.com/arviz-devs/arviz/pull/700): +319 −38
* [#701](https://github.com/arviz-devs/arviz/pull/701) (bugfix for #678): +8 −13
* [#727](https://github.com/arviz-devs/arviz/pull/727): +118 −8
* [#748](https://github.com/arviz-devs/arviz/pull/748) (bugfix for #678): +3 −2
* [#755](https://github.com/arviz-devs/arviz/pull/755): +224 −143
* [#764](https://github.com/arviz-devs/arviz/pull/764): +6 −0
* [#771](https://github.com/arviz-devs/arviz/pull/771) [UNMERGED]: +844 −0
* [#790](https://github.com/arviz-devs/arviz/pull/790) (fix for #678): +24 −24

Total lines of code added/removed from ArviZ for this section: +2356 -330

---
### Convergence Assessment

Bayesian Inference generally relies on Markov-Chain Monte Carlo samplers to aproximate numerically the posterior distributions
of interest. These methods converge to the real posterior when the number of samples tends to infinite. Thus, in all real cases
there is no guarantee (and there cannot be) that the approximated posterior is a good aproximation. Convergence assesment
algorithms try to palliate this issue by detecting bad approximations; they cannot guarantee convergence but in many cases
they can guarantee that the MCMC has not converged which is quite an improvement. ArviZ has several diagnostic functions which
serve this convergence assesment purpose. Work on this section was basically to add some new diagnostic plots following
[Vehtari et al 2019](https://arxiv.org/abs/1903.08008).

* New plotting function [`plot_ess`](https://arviz-devs.github.io/arviz/generated/arviz.plot_ess.html#arviz.plot_ess)
was created. It produces three different kinds of plot. The two first kinds `local` and
`quantile` are used to check that the MCMC has properly sampled all the parameter space; undersampling of some
region indicates convergence issues. In these two kinds, `plot_ess` allows to customize the appearance of the plot,
to show rug values for any variable in `sample_stats` or to compare the plotted values with the `mean` and `sd`
effective sample. The third kind plots the evolution of the effective sample size which should be roughly linear.
* Another plotting function [`plot_mcse`](https://arviz-devs.github.io/arviz/generated/arviz.plot_mcse.html#arviz.plot_mcse)
was also added to ArviZ. It plots the `local` or `quantile` Monte Carlo
standard error either as a scatter plot or as errorbars on the parameter values. Both kinds allow similar options
to `plot_ess` such as customizing appearence or comparing with `mean` and `sd` MCSE.

#### PRs in this caterory
* [#708](https://github.com/arviz-devs/arviz/pull/708): +405 −0
* [#724](https://github.com/arviz-devs/arviz/pull/724): +397 −37
* [#778](https://github.com/arviz-devs/arviz/pull/778) (bugfix for both #708 and #724): +4 −2

Total lines of code added/removed from ArviZ for this section: +806 -39

---
### Model Checking

In Bayesian modeling, observations are considered random variables defined by the model and some parameters. This allows
us to calculate the best fit parameters and its uncertainty and then use them to predict possible future realizations.
However, we should also check whether or not this model of random variables generates samples compatible with the observations.
This can help in detecting model limitations and finding ways to improve it. One of the algorithms available for model
checking is Leave-One-Out Probability Integral Transform (LOO-PIT) which also adds concepts from cross validation to model
checking. This algorithm was added to ArviZ along with plots to interpretate it.

* Calculation of LOO-PIT values was implemented in ArviZ using new stats function
[`az.loo_pit`](https://arviz-devs.github.io/arviz/generated/arviz.loo_pit.html#arviz.loo_pit). This function is
extremely versatile and allows to combine data form InferenceData objects with array or DataArray inputs.
As LOO-PIT uses also concepts from LOO information criterion, the defaults used for dimension order and names are the
ones defined by the work in information criteria section.
* A new plotting function
[`az.plot_loo_pit`](https://arviz-devs.github.io/arviz/generated/arviz.plot_loo_pit.html#arviz.plot_loo_pit)
was also added to plot and interpretate LOO-PIT values. It has two
ways of showing LOO-PIT values, either plotting its kernel density estimate or plotting the difference between
the empirical cumulative density function of LOO-PIT values and the exact cumulative density function of a
uniform random variable.
* Another stats function [`az.apply_test_function`](https://arviz-devs.github.io/arviz/generated/arviz.apply_test_function.html#arviz.apply_test_function) which eases the application of Bayesian test functions on
InferenceData objects was also written. This function however, as its docs explain, is generally slower than
using xarray or numpy vectorized calculations. This function is still relevant and useful for two main reasons.
The first is because it also uses `wrap_xarray_ufunc` and can then ease parallelization of test functions
calculations. The second reason is that having this function in ArviZ docs is a reminder that test functions can be
useful by themselves and that any test function can be used to perform LOO-PIT checks
on its results (see [this thread](https://discourse.mc-stan.org/t/loo-pit-algorithm-applicability/9245) for details).

#### PRs in this caterory
* [#696](https://github.com/arviz-devs/arviz/pull/696): +837 −24
* [#738](https://github.com/arviz-devs/arviz/pull/738) (documentation bugfix for #696): +2 −2
* [#781](https://github.com/arviz-devs/arviz/pull/781): +52 −4

Total lines of code added/removed from ArviZ for this section: +891 -30

---
### InferenceData scheme modifications

InferenceData objects are one key aspect of ArviZ. They provide a unified data scheme to store Bayesian inference
results from any library. They can contain the posterior, prior, prior predictive and posterior predictive samples
along with sample stats and observed data in a single object. Due to the large number of new functionalities
described in the previous three sections, we also found convenient to update the data scheme of InferenceData objects.

*  Handling of posterior predictive samples in `from_pymc3` was modified so that whenever possible posterior predictive
samples are reshaped into ArviZ default shape `chain, draw, *shape`. Moreover, when the shape of the posterior
predictive does not match the posterior shape, a warning is printed to warn the user that the posterior
predictive samples may omit whole chains.
* ArviZ example datasets `centered_eight` and `non_centered_eight` were updated. Now their posterior predictive
shape matches their prior shape.
* A `del` method was added to InferenceData objects so that groups can be deleted.
* A new group `constant_data` was added to InferenceData functions. It stores constants of the model in
addition to the observed data which was already stored in `observed_data` group. This change is still unmerged.
* Another new group was added to InferenceData objects: `log_likelihoods` group which makes ArviZ start supporting
multiple log likelihoods to be stored in a single InferenceData object. This new group is intended to store all
log likelihood data (preferably named like the variable in observed data it refers to) and the model log
probability (named `lp`). Previously, this data was stored in `sample_stats` group and only one log likelihood
could be stored. `from_dict`, `from_emcee`, `from_pymc3` and `from_pystan` have already been updated but the changes
are still unmerged.

#### PRs in this caterory
* [#702](https://github.com/arviz-devs/arviz/pull/702): +70 −17
* [#715](https://github.com/arviz-devs/arviz/pull/715): +24 −19
* [#720](https://github.com/arviz-devs/arviz/pull/720) (bugfix for #715): +2 −4
* [#739](https://github.com/arviz-devs/arviz/pull/739): +101 −9
* [#794](https://github.com/arviz-devs/arviz/pull/794) [UNMERGED]: +533 −291

Total lines of code added/removed from ArviZ for this section: +730 -340

---
### Global parameters

ArviZ functions allow a great deal of custoization but are still simple to use. This is achieved by using extensive optional
arguments in ArviZ functions each of which must have a default. Many of these defaults are shared between several
functions. This forces users to change the default explicitely on a function basis instead of giving an option to change
the default globally like matplotlib for example. Before the number of functions continues increasing making every time
more difficult to create a way of handling global options it was decided to follow matplotlibs `rcParams` to implement
an ArviZ's `rcParams` version.

* A new class to handle global options was created following matplotlib's example. This `az.rcParams` class checks for an
`arvizrc` in several locations in order to load the defaults defined there; otherwise, the defaults defined in `rcparams.py`
are used. `az.rcParams` keys cannot be modified and when modifying the value of a parameter, this value is first validated.
This two properties are key to prevent errors when functions use this `az.rcParams`. At the date of writing, 3 global
parameters have been integrated in ArviZ which will be described below. To see an always up to date list of all global
parameters in ArviZ check [`arvizrc.template`](https://github.com/arviz-devs/arviz/blob/master/arvizrc.template).
To see other parameters on the roadmap or how to add a new parameter see
[this issue](https://github.com/arviz-devs/arviz/issues/792)).
  * `data.load`: this parameter defined how is data loaded when using `from_netcdf` methods. It accepts two values: `lazy`
  to lazyly load the data using xarray and read it from disk when needed for calculations or `eager` in order to load
  the data into memory the moment the file is read.
  * `plot.max_subplots`: defines the maximum number of subplots ArviZ can add to a single figure. This prevents users to
  inadvertedly call a plotting function on too many variables which would hang up its computer.
  * `stats.information_criterion`: sets the default information criterion between `waic` and `loo`.
* A context manager, [`az.rc_context`](https://arviz-devs.github.io/arviz/generated/arviz.rc_context.html#arviz.rc_context),
was also added in order to temporarily change ArviZ defaults.
It allows either a dictionary or a file from which to read the temporal defaults as an input.

#### PRs in this caterory
* [#734](https://github.com/arviz-devs/arviz/pull/734): +374 −20
* [#787](https://github.com/arviz-devs/arviz/pull/787): +208 −37

Total lines of code added/removed from ArviZ for this section: +582 -57

---

### Other Pull Requests

Contributing to a software package is not only implementing new functions and documenting them. It is strongly
recommended to implement continuous integration builds to test everything is working properly after every single
change in the code. These test must also be maintained as they may break for external causes like a dependency
changing its API. Documentation must also be kept up to date when functions are removed. Issues posted by users should
also be addressed and so on. During my GSoC project, a small part of my work hours were dedicated to some of this
ArviZ maintainment tasks.
When looking at the number of PR in this category in this section it may seem that a lot of time was spent on this
but that is not really the case. The lines of code have been included in all sections to show that all PRs in
this category are small and quite simple. There is no changelog summary for this section as these PRs share no common
topic and they change only one or two things, thus, looking at their description is enough to see what was
changed or fixed.

* [#714](https://github.com/arviz-devs/arviz/pull/714): +1 −1
* [#723](https://github.com/arviz-devs/arviz/pull/723): +72 −17
* [#730](https://github.com/arviz-devs/arviz/pull/730): +25 −15
* [#736](https://github.com/arviz-devs/arviz/pull/736): +4 −3
* [#765](https://github.com/arviz-devs/arviz/pull/765): +76 −28
* [#768](https://github.com/arviz-devs/arviz/pull/768): +3 −1
* [#769](https://github.com/arviz-devs/arviz/pull/769): +4 −1
* [#773](https://github.com/arviz-devs/arviz/pull/773): +63 −28
* [#776](https://github.com/arviz-devs/arviz/pull/776): +0 −1
* [#779](https://github.com/arviz-devs/arviz/pull/779): +15 −1
* [#788](https://github.com/arviz-devs/arviz/pull/788): +3 −2

Total lines of code added/removed from ArviZ for this section: +266 -98



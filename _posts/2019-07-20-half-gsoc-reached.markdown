---
layout: post
title:  "Half GSoC reached"
description: "I have already reached the middle of the GSoC coding period"
date:   2019-07-15 15:10:21 +0200
mathjax: true
image: assets/images/plot_loo_pit_ecdf.png

---

Even though it feels like yesterday, the coding period started nearly a month
and a half ago. We have already been coding for more than half of the GSoC coding
period time. I have passed the first evaluation, received the first
payment and implemented or modified many functions in ArviZ (even created some bugs
too...). Therefore, it feels like a good time to review in a little depth
my work up to this point, and compare with the [timeline proposed]({{ site.url
}}/2019/04/08/write-the-proposal.html).

### Finished tasks
According to the proposed timeline, I should have finished up to task T7, and
be starting T9. I do not count T8 here because it is basically correcting bugs,
which I have already started doing and
has no finish date. In general lines, I have followed and fulfilled my initial
timeline up until here, but from here on, it will start to diverge as we have
considered more interesting to pursue some new goals.

#### T1-T3: Information Criteria
Hence, I should have finished my work on information
criteria (T1-T3: implementation, tests, documentation and examples). And
indeed I have. I have modified the API of the information criterion (IC)
stats functions `loo` and `waic` in three key ways:
* I have created a custom class `ELPDData` class so that the return value of
  IC functions in ArviZ is of this type, which results in a more meaningful
  printed text to ease interpretation of IC results.
* I have modified IC functions to use [`xarray`](http://xarray.pydata.org).
  After this change, the conversion to unlabeled array is not necessary anymore
  and pointwise IC
  values are correctly labeled, following the labels found in the log
  likelihood data from which IC is calculated.
* In addition to using `xarray` for calculations, I have also modified how the
  IC function is applied and broadcasted to each observation allowing to work
  the whole time not only with labeled data but with multidimensional data.
  Before this change, the data was reshaped _always_ to a 2D object whose
  shape was $(N_{samples}, N_{obsevations})$, now, the calculations are
  performed with objects of shape $(..., N_{samples})$ where the $...$ can be
  any shape.
Rgarding the information criteria related plots, I have added many
customization options to `plot_khat` and implementes a new plotting function
`plot_elpd`, which compares graphically pointwise IC values between 2 or more
models. In `plot_khat`, I added many new coloring and labeling options. From all
of them, the most relevant is probably the hover labels (see them below!),
which also lead me to implementing a context manager for
ipython sessions to temporarily change the backend from inline to an
interactive one.
[Check out how to use it!](https://arviz-devs.github.io/arviz/generated/arviz.interactive_backend.html)

![plot_khat]({{ site.url }}/assets/images/plot_khat.gif)

#### T4-T6: Convergence analysis tools
The other block of my proposal were convergence analysis tools. A great deal
of tools was published while I was writing the proposal, and they were added
to the stats module in ArviZ right before the start of my coding period. I
reviewed extensively the pull request that added all these tools and
read the papers to get familiar with how they worked and how
they were implemented.

In this block, my work consisted in creating plotting functions to allow
graphical exploration of all the new convergence analysis tools. These include
local, quantile and evolution effective sample size plots and local and
quantile Monte-Carlo standard error plots. Both plots leave room for great
customization, some examples can be seen below:

![plot_ess]({{ site.url }}/assets/images/plot_ess_evolution.png)

![plot_mcse]({{ site.url }}/assets/images/plot_mcse_quantile.png)

#### T#: Model checking
Moreover, I also implemented Leave One Out (LOO) Probability Integral
Transform (PIT) checks, to see if the observed data could have been generated
from the model.

LOO-PIT combines the idea of the PIT algorithm, which is that if an
observation $x_i$ is generated from a probability density function $f(x)$, then $F(x)
= P(x < x_i)$ is distributed as a uniform random variable; with the LOO
algorithm idea which is to perform the calculations on $x_i$ with the
inference results of fitting all data but $x_i$. Thus, for every $x_i$ in the
observed data, $f(x)$ is estimated
fitting all data but $x_i$ and then $x_i$ is used to check the PIT uniformity.

This kind of check requires models with a high number of observations, but can
detect problems with the model. If the LOO-PIT samples are far from
uniformity, it probably means that the observed data could never be generated
from the current model.

It also implements the comparison of the empirical cumulative density function
(ECDF)
with its theoretical value (because we know it must be uniform) and then plot
also the theoretical envelope inside of which most uniform ECDFs will end up.

![plot_loo_pit]({{ site.url }}/assets/images/plot_loo_pit_ecdf.png)

#### T7: Test on real examples
I created a github [repo](https://github.com/OriolAbril/Bayesian-Inference-examples)
to generate and store a wide range of Bayesian inference examples, and in
addition to the models in ArviZ, I also tested the functions manually on these
examples. I do not plan on stopping here however. I will check repos and
packages that use ArviZ, PyMC3 or PyStan and analyze the robustness of their
workflow. Then, I will try to apply all the functions implemented in these repos and
extend the checks performed on the model and the MCMC samples.
This could achieve two goals,
it could make these algorithms more known and it will most probably help in
finding bugs and improving the documentation.

### Future work
I have no more work planned on the convergence analysis block, but I do have
plans for the other two blocks, implementing the Leave Future Out cross
validation in the information criteria block and implementing Simulation
Based Calibration in the model checking block. Both algorithms require
refitting, which poses a huge challenge for a backend agnostic package like
ArviZ without sampling capabilities of its own. I have already started working
on some wrappers in order to allow refitting using any user defined backend
such as PyStan, PyMC3 of emcee.

In addition, I have also started working on implementing rcParams in ArviZ,
following matplotlib's implementation, in order to allow easier and better
configuration of ArviZ defaults.

Finally, of the planned tasks that lie ahead, I will not spend much time on
benchmarking (T9), because that would collide with the other ongoing GSoC project
to apply Numba to ArviZ algorithms, nor I will write a section in ArviZ
resources repo, I will probably post tutorials on how to use ArviZ's new
functions in this blog, following the example of the LOO on transformed data
example.

---
layout: post
title:  "Start of the Coding Period"
description: "Description of the first week of the Coding Period. First
impressions and experimenting with PyStan"
date:   2019-06-03 23:33:01 +0200
image: assets/images/GSoC-logo.png

---

After the admission to the GSoC program, there is a time period to get started
with the project, contact the mentors and so on. After this, the Coding Period
starts. This year, it started on May 27th. In my case, I had already
contributed to ArviZ, so I had already set up my working environment even
before the proposal submission. Thus, I dedicated this period to add detail to
my proposal and to discuss with my mentors how to tackle the different tasks.

I started by checking some papers on Bayesian Workflow and Bayesian
visualization, similar projects and going issue by issue in ArviZ to find all
feature requests related to my project. Afterwards I outlined a list of all
the methods with their priority and a possible API which was then discussed with my
mentors. I immediately started with a pull request on these methods, trying to
extend the functionality to all use cases I could think about in the most
simple and natural way (for now natural to me, once the code is reviewed and
used by other people, hopefully natural to everybody).

The first problem I came across was not having real examples on which to test
the functions implemented and modified at hand. There are many examples in
PyMC and in PyStan, however I was not familiar with their example
repositories. Hence, I eventually decided to create some custom toy examples.
This approach is obviously more time consuming than simply searching through
the archives until finding one (or more) suitable examples, but it has many
other advantages like learning to use new inference libraries to build the
examples, and using first hand all the functionality in ArviZ to assess the
predictive accuracy of the models and the convergence of the MCMC. Not only I
get to learn and I am able to test the functions I code, but I also use them
in the _same exact way_ the end users will, revealing problems that may escape
CI testing. After all, I am also an end user of ArviZ.

The first advantage and maybe the most obvious is learning different
inference libraries. I started to create examples with PyStan because it was
the one I was less familiar with, and thus, more curious about its performance
and functionalities. It is really different to Python as it is a compiled
language, and it has different types and statements too. Moreover, the
programs must be divided into some blocks. This statement may be different for
pure Python coders, but in my case, having learnt Pascal as first programming
language and having used Fortran extensively, the compiled/interpreted
language was not a significant problem. So far I have enjoyed my experience
with PyStan, being the only bad experience some issues trying to understand
error messages.

The second one may be more personal. I find that using the functions in ArviZ
where possible throughout the example creation process, simulating data and
analyzing it, has an stimulating effect on me, and I come across possible new
arguments and functionality in hopes of making the functions coded more general
and interpretable. I believe that this kind of experimentation combined with
the comments in the pull requests will result in intuitive and easy to use
functions added to ArviZ.



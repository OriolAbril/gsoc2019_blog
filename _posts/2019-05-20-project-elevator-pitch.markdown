---
layout: post
title:  "The Project"
description: "An explanation of my GSoC project for everyone. No math nor
Bayes required"
date:   2019-05-20 07:17:28 +0200
image: assets/images/ArviZ.png

---

My GSoC project is in the field of Bayesian modelling. The aim of modelling is
to help to explain, define or estimate a given aspect of the world based on
known or more accessible data. The adjective Bayesian specifies how this is
done. From a probabilistic point of view, it uses data to update our
current state of knowledge of the world. Thus, it allows to take into account
both the data available and the current knowledge on the topic. Bayesian
modelling is used in many fields, from psychology to economics or
astrophysics.

Inside Bayesian modelling, it covers mainly 2
different topics: knowing if the inference
algorithm has performed correctly and knowing which one out of several models
better predicts the quantity of interest.

### Convergence assessment

Convergence assessment stands for knowing whether or not the inference
algorithm has converged. This is important because these algorithms are proven
to work _assimptotically_, which is when the number of iterations of the
algorithm becomes infinite. However, in the real world, the number of
iterations must be finite, thus, it is crucial to know (or at least estimate)
when the algorithm is not performing properly because, in this case, the
results cannot be trusted.

### Information Criteria

Information Criteria try to evaluate the predictive accuracy of a given model
on **new** data. For example, if we used a thousand patients to create and calibrate
a model of the relation between Alzheimer and fungal infections, information
criteria estimate how accurate
will this model be when used **outside** these thousand patients. This can be
an important reason to choose when there are several models trying to explain
the same phenomenon. This is intrinsically a very difficult task which can
only expect to estimate this accuracy, never to know it, because the **new**
data is not available, however, its relevance in model comparison makes it
worth the risk.

### Why all this?

Modelling helps us understand the world around us, and this knowledge can then
be used to make our lives better; understanding the movement of electrons in
semiconductor materials allowed the creation of transistors and processors,
understanding the relation between some genes and an illness can help in its
prevention and diagnosis. Moreover, in some cases, knowing the uncertainty of
our models is also really important, if an autonomous car believes it has no
person ahead but there is large uncertainty in the measurement/prediction,
it should probably stop anyway just in case. In many of these models, the
Bayesian perspective has a clear edge. Therefore, I believe that making this
king of modelling more accessible and straightforward can have a great impact
in many aspects.

In addition, ArviZ pays a lot of attention on its documentation, explaining
and encouraging best practices in the field of Bayesian modelling, which can
be unknown and tricky due to its peculiar point of view, which can differ from
what is generally assumed as known or what is most common.

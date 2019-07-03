---
layout: post
title:  "The Proposal"
description: "Writing the proposal for ArviZ"
date:   2019-04-08 00:03:28 +0200
image: assets/images/gantt.png

---

I started my proposal following the examples in the
[Student Guide](https://google.github.io/gsocguides/student/).
They can be found
[here](https://google.github.io/gsocguides/student/proposal-example-1) and
 [here](https://google.github.io/gsocguides/student/proposal-example-2).
 To begin with, I created a Google Docs file, in order to make it easy for Arviz
 core developers to comment and review the proposal draft.

I had many problems trying to write a coherent timeline, because I was not
an expert in ArviZ nor in the algorithms involved. Thus, I had nearly no
previous relevant experience to use as "timing guide". Therefore, I found my
contributions to ArviZ -prior to writing the proposal- to be an
invaluable help when writing the timeline. I spent several days working on the
proposal alone and once I was relatively happy with the draft I shared it with
the ArviZ community.

After receiving the comments of the possible mentors, I started paying more
attention to the format of the proposal. I created a
[GitHub repo](https://github.com/OriolAbril/gsoc2019) to continue developing
the proposal, now following the format of the
[NumFOCUS template](https://github.com/numfocus/gsoc/blob/master/templates/proposal.md).
The generated proposal follows quite closely the template except in the timeline
section.

I felt that the timeline explained as a list of tasks with some planned start
and end date did not provide a good idea of the time dedicated to each task nor
the relations between them. Therefore I decided to use a Gantt chart to make the
timeline more visual and easy to follow.

<img src="{{ '/assets/images/gantt.png' | prepend: site.baseurl | prepend: site.url }}" alt="Gantt Chart" class="image center">

This Gantt chart was built with matplotlib. It allows the reader to distinguish
clearly between the different task types as well as their length and relations.
Moreover, it shows the key dates of the program which can be identified at first
glance. The explanation of the labels can be found in the
[complete proposal](https://github.com/OriolAbril/gsoc2019/blob/master/proposal.pdf). Below there is the code used to generate the Gantt Chart:

{% highlight python %}
{% include_relative  code/gantt.py %}
{% endhighlight %}

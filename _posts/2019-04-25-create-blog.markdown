---
layout: post
title:  "The Blog"
description: "Creating this website to blog about my GSoC experience"
date:   2019-04-25 00:22:17 +0200
image: assets/images/jekyll.png

---

One of the requisites of participating in GSoC with NumFOCUS is to blog about the experience.
To this end, they provide a [guide](https://github.com/numfocus/gsoc/blob/master/gsoc_student_blog_setup.md)
that helps setting up the website with [Jekyll](https://jekyllrb.com/) and [GitHub pages](https://pages.github.com/).

The first step was to build the blog locally using Jekyll and Ruby, following thw guide provided by NumFOCUS.
By default, the theme used was [minima](https://github.com/jekyll/minima), which runs smoothly without
any extra configuration. In my case though, I decided to use a custom theme. Moreover, I decided to use a theme not
supported by GitHub pages. The main reason was to practice customizing themes with the long term goal in mind to use
a custom theme for a future personal website. Using a theme not supported by GitHub pages obiously implies more work
but, in my opinion, it was worth it.

I had heard about the [HTML5 UP](https://html5up.net/) themes, and after some searching, I found a port to Jekyll of
one of the themes I was interested in: [Forty](https://github.com/andrewbanchich/forty-jekyll-theme). Thus, I decided
to use this theme for my blog. I started customizing the layouts and templates locally checking the build with the
command `bundle exec jekyll serve`. It was my first conding experience with Ruby and the second time using html,
and some thing stroke me as odd. The main unexpected behaviour was the fact that using `jekyll build` and opening
the _index.html_ file did not show the website properly, even when `jekyll serve` did. I also needed some time to
get used to [Liquid](https://jekyllrb.com/docs/liquid/) mainly because it works with both html and markdown but not
all markdown is properly converted when building the site and this gave me some headaches.

The next step was to publish the website using GitHub pages. Creating the repository and so on had no difficulties,
whereas building the website in GitHub did. This however may not be indicative of the difficulty of the two tasks
because I have some experience with GitHub, managing repositories and Git, whereas it was my first contact with Jekyll,
Ruby and building a website without sphinx. Eventually, I found out/accepted that GitHub pages has limited support for
external Jekyll and Ruby plugins, but if their source files are included in the repository, things are bound to work
sooner or later.

Overall, I am really happy with my result and I will surely use GitHub pages to host more websites due to its ease of use
while allowing complete customization too. I still feel like I spend an execcive ammout of time for every single edit, but
little by little I am starting to feel more comfortable with it. In additon, once the template has been created, the
work is basically writing text posts, which can be written in Markdown.

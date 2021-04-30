---
title: Convex Hulls, Delaunay Triangulations and Voronoi Diagrams
author: Indranil Ghosh
date: '2021-04-30'
slug: computational-geometry
categories: ["Python", "visualization", "Computational Geometry", "spatial algorithms"]
tags: ["statsbomb api", "scipy.spatial", "scipy", "sdpatial algorithms"]
subtitle: ''
summary: ''
authors: []
lastmod: '2021-04-30T10:46:23+05:30'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

In our [last tutorial](https://realsoccerexpand.netlify.app/post/pass-network-analysis/), we studied how to visualize a pass network for the teams from a particular match and how to analyse the networks using knowledge from complex network analysis literature. In this tutorial we will learn how to implement ideas from computational geometry on football spatial event or tracking data and generate important visualizations and analyses. This post is highly influenced by the concepts discussed in the second chapter *How Slime Moulds Built Barcelona* from the book [**Soccermatics**](https://www.amazon.in/Soccermatics-Mathematical-Adventures-Beautiful-Bloomsbury-ebook/dp/B01AIB7YKE).

First we will study how to develop a *convex hull* around those points (locations denoted by x- and y- coordinates) from where a player had made a pass in a particular game. Mathematically, if these points are contained in a set `**X**` then the *convex hull* is the smallest convex set that contains `**X**`. This will help us get an idea about the optimal field coverage of a player during the match. Let us see how a convex hull for a set of points looks like:

![](convexhull.png)
This figure has been adapted from the [wikipedia article](https://en.wikipedia.org/wiki/Convex_hull#:~:text=In%20mathematics%2C%20the%20convex%20hull%20or%20convex%20envelope,is%20the%20smallest%20convex%20set%20that%20contains%20X.) on *convex hulls*.

Now we will collect the event data from a particular match

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

First we will study how to develop a *convex hull* around those points (locations denoted by x- and y- coordinates) from where a player had made a pass or had taken a shot in a particular game. Mathematically, if these points are contained in a set **`X`** then the *convex hull* is the smallest convex set that contains **`X`**. This will help us get an idea about the optimal field coverage of a player during the match. Let us see how a convex hull for a set of points looks like:

![](convexhull.png)
This figure has been adapted from the [wikipedia article](https://en.wikipedia.org/wiki/Convex_hull#:~:text=In%20mathematics%2C%20the%20convex%20hull%20or%20convex%20envelope,is%20the%20smallest%20convex%20set%20that%20contains%20X.) on *convex hulls*.

Before we start with our data collection and analysis we need to download the [`scipy`](https://www.scipy.org/) package which provides us with a collection of modules for working on scientific computation with Python.
Let us `pip` install the package:


```python
pip install scipy
```

We will first import all the packages that we need in this tutorial:


```python
from statsbombpy import sb # statsbomb api
import matplotlib.pyplot as plt # matplotlib for plotting
from mplsoccer.pitch import Pitch # for drawing the football pitch
import seaborn as sns # seaborn for plotting useful statistical graphs
import numpy as np # numerical python package
import pandas as pd # pandas for manipulating and analysing data
import networkx as nx # package for complex network analysis
```

Further, for this post, we need to use the `scipy.spatial` module that allows us to work with spatial algorithms and data structures. As we are going to work with *convex hulls* first, let us import the `ConvexHull` classes from `scipy.spatial`: 


```python
from scipy.spatial import ConvexHull
```

Next, we will collect the event data from a particular match and filter the data in such a way that the event `type` will be set to `Pass` or `Shot`, fetching us all the data for pass and shot events. 


```python
comp = sb.competitions()
```

```
## credentials were not supplied. open data access only
```

```python
print(comp.to_markdown())
```

```
## |    |   competition_id |   season_id | country_name             | competition_name        | competition_gender   | season_name   | match_updated              | match_available            |
## |---:|-----------------:|------------:|:-------------------------|:------------------------|:---------------------|:--------------|:---------------------------|:---------------------------|
## |  0 |               16 |           4 | Europe                   | Champions League        | male                 | 2018/2019     | 2021-04-19T17:36:05.724116 | 2021-04-19T17:36:05.724116 |
## |  1 |               16 |           1 | Europe                   | Champions League        | male                 | 2017/2018     | 2021-01-23T21:55:30.425330 | 2021-01-23T21:55:30.425330 |
## |  2 |               16 |           2 | Europe                   | Champions League        | male                 | 2016/2017     | 2020-08-26T12:33:15.869622 | 2020-07-29T05:00           |
## |  3 |               16 |          27 | Europe                   | Champions League        | male                 | 2015/2016     | 2020-08-26T12:33:15.869622 | 2020-07-29T05:00           |
## |  4 |               16 |          26 | Europe                   | Champions League        | male                 | 2014/2015     | 2020-08-26T12:33:15.869622 | 2020-07-29T05:00           |
## |  5 |               16 |          25 | Europe                   | Champions League        | male                 | 2013/2014     | 2020-08-26T12:33:15.869622 | 2020-07-29T05:00           |
## |  6 |               16 |          24 | Europe                   | Champions League        | male                 | 2012/2013     | 2020-08-26T12:33:15.869622 | 2020-07-29T05:00           |
## |  7 |               16 |          23 | Europe                   | Champions League        | male                 | 2011/2012     | 2020-08-26T12:33:15.869622 | 2020-07-29T05:00           |
## |  8 |               16 |          22 | Europe                   | Champions League        | male                 | 2010/2011     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## |  9 |               16 |          21 | Europe                   | Champions League        | male                 | 2009/2010     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 10 |               16 |          41 | Europe                   | Champions League        | male                 | 2008/2009     | 2020-08-30T10:18:39.435424 | 2020-08-30T10:18:39.435424 |
## | 11 |               16 |          39 | Europe                   | Champions League        | male                 | 2006/2007     | 2021-03-31T04:18:30.437060 | 2021-03-31T04:18:30.437060 |
## | 12 |               16 |          37 | Europe                   | Champions League        | male                 | 2004/2005     | 2021-04-01T06:18:57.459032 | 2021-04-01T06:18:57.459032 |
## | 13 |               16 |          44 | Europe                   | Champions League        | male                 | 2003/2004     | 2021-04-01T00:34:59.472485 | 2021-04-01T00:34:59.472485 |
## | 14 |               16 |          76 | Europe                   | Champions League        | male                 | 1999/2000     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 15 |               37 |          42 | England                  | FA Women's Super League | female               | 2019/2020     | 2021-04-28T19:48:01.172671 | 2021-04-28T19:48:01.172671 |
## | 16 |               37 |           4 | England                  | FA Women's Super League | female               | 2018/2019     | 2021-04-28T19:48:01.166958 | 2021-04-28T19:48:01.166958 |
## | 17 |               43 |           3 | International            | FIFA World Cup          | male                 | 2018          | 2020-10-25T14:03:50.263266 | 2020-10-25T14:03:50.263266 |
## | 18 |               11 |          42 | Spain                    | La Liga                 | male                 | 2019/2020     | 2020-12-18T12:10:38.985394 | 2020-12-18T12:10:38.985394 |
## | 19 |               11 |           4 | Spain                    | La Liga                 | male                 | 2018/2019     | 2021-04-20T03:24:51.029365 | 2021-04-20T03:24:51.029365 |
## | 20 |               11 |           1 | Spain                    | La Liga                 | male                 | 2017/2018     | 2021-04-19T17:36:05.805404 | 2021-04-19T17:36:05.805404 |
## | 21 |               11 |           2 | Spain                    | La Liga                 | male                 | 2016/2017     | 2021-02-02T23:24:58.985975 | 2021-02-02T23:24:58.985975 |
## | 22 |               11 |          27 | Spain                    | La Liga                 | male                 | 2015/2016     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 23 |               11 |          26 | Spain                    | La Liga                 | male                 | 2014/2015     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 24 |               11 |          25 | Spain                    | La Liga                 | male                 | 2013/2014     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 25 |               11 |          24 | Spain                    | La Liga                 | male                 | 2012/2013     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 26 |               11 |          23 | Spain                    | La Liga                 | male                 | 2011/2012     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 27 |               11 |          22 | Spain                    | La Liga                 | male                 | 2010/2011     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 28 |               11 |          21 | Spain                    | La Liga                 | male                 | 2009/2010     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 29 |               11 |          41 | Spain                    | La Liga                 | male                 | 2008/2009     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 30 |               11 |          40 | Spain                    | La Liga                 | male                 | 2007/2008     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 31 |               11 |          39 | Spain                    | La Liga                 | male                 | 2006/2007     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 32 |               11 |          38 | Spain                    | La Liga                 | male                 | 2005/2006     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 33 |               11 |          37 | Spain                    | La Liga                 | male                 | 2004/2005     | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 34 |               49 |           3 | United States of America | NWSL                    | female               | 2018          | 2020-07-29T05:00           | 2020-07-29T05:00           |
## | 35 |                2 |          44 | England                  | Premier League          | male                 | 2003/2004     | 2020-08-31T20:40:28.969635 | 2020-08-31T20:40:28.969635 |
## | 36 |               72 |          30 | International            | Women's World Cup       | female               | 2019          | 2020-07-29T05:00           | 2020-07-29T05:00           |
```

Next, let us decide on a `competition_id` and a `season_id` to extract matches from a particular season and competition:


```python
mat = sb.matches(competition_id = 11, season_id = 1)
```

```
## credentials were not supplied. open data access only
```

```python
print(mat.to_markdown())
```

```
## |    |   match_id | match_date   | kick_off     | competition     | season    | home_team           | away_team           |   home_score |   away_score | match_status   | match_status_360   | last_updated     | last_updated_360   |   match_week | competition_stage   | stadium                       | referee           | data_version   |   shot_fidelity_version | xy_fidelity_version   |
## |---:|-----------:|:-------------|:-------------|:----------------|:----------|:--------------------|:--------------------|-------------:|-------------:|:---------------|:-------------------|:-----------------|:-------------------|-------------:|:--------------------|:------------------------------|:------------------|:---------------|------------------------:|:----------------------|
## |  0 |       9592 | 2017-09-09   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Espanyol            |            5 |            0 | available      | unscheduled        | 2020-07-29T05:00 |                    |            3 | Regular Season      | Camp Nou                      | Jesús Gil         | 1.1.0          |                       2 |                       |
## |  1 |       9870 | 2018-04-07   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Leganés             |            3 |            1 | available      | unscheduled        | 2020-07-29T05:00 |                    |           31 | Regular Season      | Camp Nou                      | nan               | 1.0.2          |                       2 |                       |
## |  2 |       9783 | 2018-02-04   | 16:15:00.000 | Spain - La Liga | 2017/2018 | Espanyol            | Barcelona           |            1 |            1 | available      | unscheduled        | 2020-07-29T05:00 |                    |           22 | Regular Season      | RCDE Stadium                  | Jesús Gil         | 1.0.2          |                       2 |                       |
## |  3 |       9700 | 2017-12-02   | 13:00:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Celta Vigo          |            2 |            2 | available      | unscheduled        | 2020-07-29T05:00 |                    |           14 | Regular Season      | Camp Nou                      | M. Melero         | 1.0.2          |                       2 |                       |
## |  4 |       9860 | 2018-03-31   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Sevilla             | Barcelona           |            2 |            2 | available      | unscheduled        | 2020-07-29T05:00 |                    |           30 | Regular Season      | Estadio Ramón Sánchez Pizjuán | José González     | 1.0.2          |                       2 |                       |
## |  5 |       9695 | 2017-11-26   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Valencia            | Barcelona           |            1 |            1 | available      | unscheduled        | 2020-07-29T05:00 |                    |           13 | Regular Season      | Estadio de Mestalla           | Ignacio Iglesias  | 1.0.2          |                       2 |                       |
## |  6 |       9794 | 2018-02-11   | 16:15:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Getafe              |            0 |            0 | available      | unscheduled        | 2020-07-29T05:00 |                    |           23 | Regular Season      | Camp Nou                      | David Fernández   | 1.0.2          |                       2 |                       |
## |  7 |       9717 | 2017-12-10   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Villarreal          | Barcelona           |            0 |            2 | available      | unscheduled        | 2020-07-29T05:00 |                    |           15 | Regular Season      | Estadio de la Cerámica        | R. De Burgos      | 1.0.2          |                       2 |                       |
## |  8 |       9673 | 2017-11-04   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Sevilla             |            2 |            1 | available      | unscheduled        | 2020-07-29T05:00 |                    |           11 | Regular Season      | Camp Nou                      | José González     | 1.0.2          |                       2 |                       |
## |  9 |       9650 | 2017-10-21   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Málaga              |            2 |            0 | available      | unscheduled        | 2020-07-29T05:00 |                    |            9 | Regular Season      | Camp Nou                      | P. González       | 1.0.2          |                       2 |                       |
## | 10 |       9620 | 2017-09-23   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Girona              | Barcelona           |            0 |            3 | available      | unscheduled        | 2020-07-29T05:00 |                    |            6 | Regular Season      | Estadi Municipal de Montilivi | nan               | 1.0.2          |                       2 |                       |
## | 11 |       9827 | 2018-03-01   | 21:00:00.000 | Spain - La Liga | 2017/2018 | Las Palmas          | Barcelona           |            1 |            1 | available      | unscheduled        | 2020-07-29T05:00 |                    |           26 | Regular Season      | Estadio de Gran Canaria       | Antonio Mateu     | 1.0.2          |                       2 |                       |
## | 12 |       9837 | 2018-03-04   | 16:15:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Atlético Madrid     |            1 |            0 | available      | unscheduled        | 2020-07-29T05:00 |                    |           27 | Regular Season      | Camp Nou                      | Jesús Gil         | 1.0.2          |                       2 |                       |
## | 13 |       9912 | 2018-04-29   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Deportivo La Coruna | Barcelona           |            2 |            4 | available      | unscheduled        | 2020-07-29T05:00 |                    |           35 | Regular Season      | Estadio Abanca-Riazor         | R. De Burgos      | 1.0.2          |                       2 |                       |
## | 14 |       9811 | 2018-02-24   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Girona              |            6 |            1 | available      | unscheduled        | 2020-07-29T05:00 |                    |           25 | Regular Season      | Camp Nou                      | J. Alberola Rojas | 1.0.2          |                       2 |                       |
## | 15 |       9642 | 2017-10-14   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Atlético Madrid     | Barcelona           |            1 |            1 | available      | unscheduled        | 2020-07-29T05:00 |                    |            8 | Regular Season      | Estadio Wanda Metropolitano   | Antonio Mateu     | 1.0.2          |                       2 |                       |
## | 16 |       9774 | 2018-01-28   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Deportivo Alavés    |            2 |            1 | available      | unscheduled        | 2020-07-29T05:00 |                    |           21 | Regular Season      | Camp Nou                      | Ignacio Iglesias  | 1.0.2          |                       2 |                       |
## | 17 |       9602 | 2017-09-16   | 16:15:00.000 | Spain - La Liga | 2017/2018 | Getafe              | Barcelona           |            1 |            2 | available      | unscheduled        | 2020-07-29T05:00 |                    |            4 | Regular Season      | Coliseum Alfonso Pérez        | David Fernández   | 1.0.2          |                       2 |                       |
## | 18 |       9948 | 2018-05-20   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Real Sociedad       |            1 |            0 | available      | unscheduled        | 2020-07-29T05:00 |                    |           38 | Regular Season      | Camp Nou                      | J. Munuera        | 1.0.2          |                       2 |                       |
## | 19 |       9682 | 2017-11-18   | 16:15:00.000 | Spain - La Liga | 2017/2018 | Leganés             | Barcelona           |            0 |            3 | available      | unscheduled        | 2020-07-29T05:00 |                    |           12 | Regular Season      | Estadio Municipal de Butarque | Alberto Undiano   | 1.0.2          |                       2 |                       |
## | 20 |       9581 | 2017-08-26   | 18:15:00.000 | Spain - La Liga | 2017/2018 | Deportivo Alavés    | Barcelona           |            0 |            2 | available      | unscheduled        | 2020-07-29T05:00 |                    |            2 | Regular Season      | Estadio de Mendizorroza       | nan               | 1.0.2          |                       2 |                       |
## | 21 |       9726 | 2017-12-17   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Deportivo La Coruna |            4 |            0 | available      | unscheduled        | 2020-07-29T05:00 |                    |           16 | Regular Season      | Camp Nou                      | Antonio Mateu     | 1.0.2          |                       2 |                       |
## | 22 |       9754 | 2018-01-14   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Real Sociedad       | Barcelona           |            2 |            4 | available      | unscheduled        | 2020-07-29T05:00 |                    |           19 | Regular Season      | Reale Arena                   | José González     | 1.1.0          |                       2 |                       |
## | 23 |       9575 | 2017-08-20   | 20:15:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Real Betis          |            2 |            0 | available      | unscheduled        | 2020-07-29T05:00 |                    |            1 | Regular Season      | Camp Nou                      | nan               | 1.1.0          |                       2 |                       |
## | 24 |       9765 | 2018-01-21   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Real Betis          | Barcelona           |            0 |            5 | available      | unscheduled        | 2020-07-29T05:00 |                    |           20 | Regular Season      | Estadio Benito Villamarín     | S. Jaime          | 1.1.0          |                       2 |                       |
## | 25 |       9889 | 2018-04-17   | 21:00:00.000 | Spain - La Liga | 2017/2018 | Celta Vigo          | Barcelona           |            2 |            2 | available      | unscheduled        | 2020-07-29T05:00 |                    |           33 | Regular Season      | Abanca-Balaídos               | David Fernández   | 1.1.0          |                       2 |                       |
## | 26 |       9928 | 2018-05-09   | 20:00:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Villarreal          |            5 |            1 | available      | unscheduled        | 2020-07-29T05:00 |                    |           34 | Regular Season      | Camp Nou                      | nan               | 1.1.0          |                       2 |                       |
## | 27 |       9609 | 2017-09-19   | 22:00:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Eibar               |            6 |            1 | available      | unscheduled        | 2020-07-29T05:00 |                    |            5 | Regular Season      | Camp Nou                      | nan               | 1.0.2          |                       2 |                       |
## | 28 |       9636 | 2017-10-01   | 16:15:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Las Palmas          |            3 |            0 | available      | unscheduled        | 2020-07-29T05:00 |                    |            7 | Regular Season      | Camp Nou                      | nan               | 1.0.2          |                       2 |                       |
## | 29 |       9661 | 2017-10-28   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Athletic Bilbao     | Barcelona           |            0 |            2 | available      | unscheduled        | 2020-07-29T05:00 |                    |           10 | Regular Season      | San Mamés Barria              | nan               | 1.0.2          |                       2 |                       |
## | 30 |       9736 | 2017-12-23   | 13:00:00.000 | Spain - La Liga | 2017/2018 | Real Madrid         | Barcelona           |            0 |            3 | available      | unscheduled        | 2020-07-29T05:00 |                    |           17 | Regular Season      | Estadio Santiago Bernabéu     | nan               | 1.1.0          |                       2 |                       |
## | 31 |       9742 | 2018-01-07   | 16:15:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Levante             |            3 |            0 | available      | unscheduled        | 2020-07-29T05:00 |                    |           18 | Regular Season      | Camp Nou                      | nan               | 1.0.2          |                       2 |                       |
## | 32 |       9799 | 2018-02-17   | 16:15:00.000 | Spain - La Liga | 2017/2018 | Eibar               | Barcelona           |            0 |            2 | available      | unscheduled        | 2020-07-29T05:00 |                    |           24 | Regular Season      | Estadio Municipal de Ipurúa   | nan               | 1.0.2          |                       2 |                       |
## | 33 |       9855 | 2018-03-18   | 16:15:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Athletic Bilbao     |            2 |            0 | available      | unscheduled        | 2020-07-29T05:00 |                    |           29 | Regular Season      | Camp Nou                      | nan               | 1.0.2          |                       2 |                       |
## | 34 |       9880 | 2018-04-14   | 16:15:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Valencia            |            2 |            1 | available      | unscheduled        | 2020-07-29T05:00 |                    |           32 | Regular Season      | Camp Nou                      | nan               | 1.0.2          |                       2 |                       |
## | 35 |       9924 | 2018-05-06   | 20:45:00.000 | Spain - La Liga | 2017/2018 | Barcelona           | Real Madrid         |            2 |            2 | available      | unscheduled        | 2020-07-29T05:00 |                    |           36 | Regular Season      | Camp Nou                      | nan               | 1.0.2          |                       2 |                       |
```

So, we have extracted the matches from 2017-18 La Liga season. Now let us decide on a particular match to extract the event data from:


```python
events = sb.events(match_id = 9609)
```

```
## credentials were not supplied. open data access only
```

```python
print(events.head(10).to_markdown())
```

```
## |    |   bad_behaviour_card |   ball_receipt_outcome |   ball_recovery_recovery_failure |   block_deflection |   carry_end_location |   clearance_aerial_won |   counterpress |   dribble_outcome |   dribble_overrun |   duel_outcome |   duel_type |   duration |   foul_committed_advantage |   foul_committed_penalty |   foul_committed_type |   foul_won_advantage |   foul_won_defensive |   foul_won_penalty |   goalkeeper_body_part |   goalkeeper_end_location |   goalkeeper_outcome |   goalkeeper_position |   goalkeeper_technique |   goalkeeper_type | id                                   |   index |   interception_outcome | location     |   match_id |   minute |   pass_aerial_won |   pass_angle |   pass_assisted_shot_id |   pass_backheel | pass_body_part   |   pass_cross |   pass_cut_back |   pass_deflected | pass_end_location   |   pass_goal_assist | pass_height   |   pass_length |   pass_outcome | pass_recipient              |   pass_shot_assist |   pass_switch | pass_type   |   period | play_pattern   | player                         | position             |   possession | possession_team   | related_events                           |   second |   shot_aerial_won |   shot_body_part |   shot_end_location |   shot_freeze_frame |   shot_key_pass_id |   shot_one_on_one |   shot_outcome |   shot_redirect |   shot_statsbomb_xg |   shot_technique |   shot_type |   substitution_outcome |   substitution_replacement | tactics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | team      | timestamp    | type        |   under_pressure |
## |---:|---------------------:|-----------------------:|---------------------------------:|-------------------:|---------------------:|-----------------------:|---------------:|------------------:|------------------:|---------------:|------------:|-----------:|---------------------------:|-------------------------:|----------------------:|---------------------:|---------------------:|-------------------:|-----------------------:|--------------------------:|---------------------:|----------------------:|-----------------------:|------------------:|:-------------------------------------|--------:|-----------------------:|:-------------|-----------:|---------:|------------------:|-------------:|------------------------:|----------------:|:-----------------|-------------:|----------------:|-----------------:|:--------------------|-------------------:|:--------------|--------------:|---------------:|:----------------------------|-------------------:|--------------:|:------------|---------:|:---------------|:-------------------------------|:---------------------|-------------:|:------------------|:-----------------------------------------|---------:|------------------:|-----------------:|--------------------:|--------------------:|-------------------:|------------------:|---------------:|----------------:|--------------------:|-----------------:|------------:|-----------------------:|---------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|:-------------|:------------|-----------------:|
## |  0 |                  nan |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |      0     |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | 2fc157ab-bb2c-48f7-a099-45411bc6d7db |       1 |                    nan | nan          |       9609 |        0 |               nan |   nan        |                     nan |             nan | nan              |          nan |             nan |              nan | nan                 |                nan | nan           |      nan      |            nan | nan                         |                nan |           nan | nan         |        1 | Regular Play   | nan                            | nan                  |            1 | Barcelona         | nan                                      |        0 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | {'formation': 4321, 'lineup': [{'player': {'id': 20055, 'name': 'Marc-André ter Stegen'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 6374, 'name': 'Nélson Cabral Semedo'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 2}, {'player': {'id': 5213, 'name': 'Gerard Piqué Bernabéu'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 3}, {'player': {'id': 5506, 'name': 'Javier Alejandro Mascherano'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 14}, {'player': {'id': 6849, 'name': 'Lucas Digne'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 19}, {'player': {'id': 5542, 'name': 'José Paulo Bezzera Maciel Júnior'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 15}, {'player': {'id': 5203, 'name': 'Sergio Busquets i Burgos'}, 'position': {'id': 14, 'name': 'Center Midfield'}, 'jersey_number': 5}, {'player': {'id': 5216, 'name': 'Andrés Iniesta Luján'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 8}, {'player': {'id': 3726, 'name': 'Gerard Deulofeu Lázaro'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'jersey_number': 16}, {'player': {'id': 6609, 'name': 'Denis Suárez Fernández'}, 'position': {'id': 21, 'name': 'Left Wing'}, 'jersey_number': 6}, {'player': {'id': 5503, 'name': 'Lionel Andrés Messi Cuccittini'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'jersey_number': 10}]} | Barcelona | 00:00:00.000 | Starting XI |              nan |
## |  1 |                  nan |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |      0     |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | 57eb014c-03f7-4f4d-a759-1515ea7d97b3 |       2 |                    nan | nan          |       9609 |        0 |               nan |   nan        |                     nan |             nan | nan              |          nan |             nan |              nan | nan                 |                nan | nan           |      nan      |            nan | nan                         |                nan |           nan | nan         |        1 | Regular Play   | nan                            | nan                  |            1 | Barcelona         | nan                                      |        0 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | {'formation': 4231, 'lineup': [{'player': {'id': 6698, 'name': 'Marko Dmitrović'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 25}, {'player': {'id': 6709, 'name': 'Anaitz Arbilla Zabala'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 18}, {'player': {'id': 6708, 'name': 'Paulo André Rodrigues de Oliveira'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 12}, {'player': {'id': 6924, 'name': 'Alejandro Gálvez Jimena'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 3}, {'player': {'id': 6776, 'name': 'David Juncà Reñé'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 23}, {'player': {'id': 6775, 'name': 'Daniel García Carrillo'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 14}, {'player': {'id': 6712, 'name': 'Gonzalo Escalante'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 5}, {'player': {'id': 6699, 'name': 'Ander Capa Rodríguez'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'jersey_number': 7}, {'player': {'id': 6701, 'name': 'Joan Jordán Moreno'}, 'position': {'id': 19, 'name': 'Center Attacking Midfield'}, 'jersey_number': 24}, {'player': {'id': 5687, 'name': 'Takashi Inui'}, 'position': {'id': 21, 'name': 'Left Wing'}, 'jersey_number': 8}, {'player': {'id': 6700, 'name': 'Sergio Enrich Ametller'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'jersey_number': 9}]}                      | Eibar     | 00:00:00.000 | Starting XI |              nan |
## |  2 |                  nan |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |      0     |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | 94e0c664-c1e9-4ee7-881e-d81124f3836a |       3 |                    nan | nan          |       9609 |        0 |               nan |   nan        |                     nan |             nan | nan              |          nan |             nan |              nan | nan                 |                nan | nan           |      nan      |            nan | nan                         |                nan |           nan | nan         |        1 | Regular Play   | nan                            | nan                  |            1 | Barcelona         | ['55c6967a-18cc-4ff8-a4be-d0e3915fc22d'] |        0 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Eibar     | 00:00:00.000 | Half Start  |              nan |
## |  3 |                  nan |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |      0     |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | 55c6967a-18cc-4ff8-a4be-d0e3915fc22d |       4 |                    nan | nan          |       9609 |        0 |               nan |   nan        |                     nan |             nan | nan              |          nan |             nan |              nan | nan                 |                nan | nan           |      nan      |            nan | nan                         |                nan |           nan | nan         |        1 | Regular Play   | nan                            | nan                  |            1 | Barcelona         | ['94e0c664-c1e9-4ee7-881e-d81124f3836a'] |        0 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Barcelona | 00:00:00.000 | Half Start  |              nan |
## |  4 |                  nan |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |      0     |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | aa8d3682-d03c-4948-99c5-d9120d8d0555 |    1850 |                    nan | nan          |       9609 |       45 |               nan |   nan        |                     nan |             nan | nan              |          nan |             nan |              nan | nan                 |                nan | nan           |      nan      |            nan | nan                         |                nan |           nan | nan         |        2 | Regular Play   | nan                            | nan                  |           92 | Eibar             | ['bb6b9211-45dd-4305-b0a3-d0244deeab04'] |        0 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Barcelona | 00:00:00.000 | Half Start  |              nan |
## |  5 |                  nan |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |      0     |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | bb6b9211-45dd-4305-b0a3-d0244deeab04 |    1851 |                    nan | nan          |       9609 |       45 |               nan |   nan        |                     nan |             nan | nan              |          nan |             nan |              nan | nan                 |                nan | nan           |      nan      |            nan | nan                         |                nan |           nan | nan         |        2 | Regular Play   | nan                            | nan                  |           92 | Eibar             | ['aa8d3682-d03c-4948-99c5-d9120d8d0555'] |        0 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Eibar     | 00:00:00.000 | Half Start  |              nan |
## |  6 |                  nan |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |      1.07  |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | 3f6b644f-5ce2-4b89-a912-bfbc05475cbb |       5 |                    nan | [60.0, 40.0] |       9609 |        0 |               nan |    -2.79282  |                     nan |             nan | Left Foot        |          nan |             nan |              nan | [49.0, 36.0]        |                nan | Ground Pass   |       11.7047 |            nan | Sergio Busquets i Burgos    |                nan |           nan | Kick Off    |        1 | From Kick Off  | Lionel Andrés Messi Cuccittini | Center Forward       |            2 | Barcelona         | ['b8523951-7a5b-4304-a768-cca7c751a9be'] |        0 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Barcelona | 00:00:00.480 | Pass        |              nan |
## |  7 |                  nan |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |      0.924 |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | 8c147c94-859d-446a-9bdf-1d2cd2307007 |       8 |                    nan | [46.0, 36.0] |       9609 |        0 |               nan |    -0.628796 |                     nan |             nan | Right Foot       |          nan |             nan |              nan | [57.0, 28.0]        |                nan | Ground Pass   |       13.6015 |            nan | Andrés Iniesta Luján        |                nan |           nan | nan         |        1 | From Kick Off  | Sergio Busquets i Burgos       | Center Midfield      |            2 | Barcelona         | ['1e11a8d0-7beb-446c-9352-7f336b1703b5'] |        3 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Barcelona | 00:00:03.252 | Pass        |              nan |
## |  8 |                  nan |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |      1.48  |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | e7a83277-0098-460b-9116-4550784f5382 |      11 |                    nan | [56.0, 28.0] |       9609 |        0 |               nan |    -3.02216  |                     nan |             nan | Right Foot       |          nan |             nan |              nan | [31.0, 25.0]        |                nan | Ground Pass   |       25.1794 |            nan | Javier Alejandro Mascherano |                nan |           nan | nan         |        1 | From Kick Off  | Andrés Iniesta Luján           | Left Center Midfield |            2 | Barcelona         | ['5a1cf3a1-6c31-4780-9cb0-84af87189db6'] |        4 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Barcelona | 00:00:04.213 | Pass        |              nan |
## |  9 |                  nan |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |      1.379 |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | 32ff139c-0932-48c1-9ab5-b392cde31db0 |      14 |                    nan | [31.0, 27.0] |       9609 |        0 |               nan |     1.72945  |                     nan |             nan | Right Foot       |          nan |             nan |              nan | [27.0, 52.0]        |                nan | Ground Pass   |       25.318  |            nan | Gerard Piqué Bernabéu       |                nan |           nan | nan         |        1 | From Kick Off  | Javier Alejandro Mascherano    | Left Center Back     |            2 | Barcelona         | ['88bfdb69-91a4-491c-8866-f49ec1b701d1'] |        7 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Barcelona | 00:00:07.225 | Pass        |              nan |
```

```python
print(events.tail(10).to_markdown())
```

```
## |      | bad_behaviour_card   |   ball_receipt_outcome |   ball_recovery_recovery_failure |   block_deflection |   carry_end_location |   clearance_aerial_won |   counterpress |   dribble_outcome |   dribble_overrun |   duel_outcome |   duel_type |   duration |   foul_committed_advantage |   foul_committed_penalty |   foul_committed_type |   foul_won_advantage |   foul_won_defensive |   foul_won_penalty |   goalkeeper_body_part |   goalkeeper_end_location |   goalkeeper_outcome |   goalkeeper_position |   goalkeeper_technique |   goalkeeper_type | id                                   |   index |   interception_outcome | location     |   match_id |   minute |   pass_aerial_won |   pass_angle |   pass_assisted_shot_id |   pass_backheel |   pass_body_part |   pass_cross |   pass_cut_back |   pass_deflected |   pass_end_location |   pass_goal_assist |   pass_height |   pass_length |   pass_outcome |   pass_recipient |   pass_shot_assist |   pass_switch |   pass_type |   period | play_pattern   | player                   | position              |   possession | possession_team   | related_events                           |   second |   shot_aerial_won |   shot_body_part |   shot_end_location |   shot_freeze_frame |   shot_key_pass_id |   shot_one_on_one |   shot_outcome |   shot_redirect |   shot_statsbomb_xg |   shot_technique |   shot_type | substitution_outcome   | substitution_replacement         |   tactics | team      | timestamp    | type          |   under_pressure |
## |-----:|:---------------------|-----------------------:|---------------------------------:|-------------------:|---------------------:|-----------------------:|---------------:|------------------:|------------------:|---------------:|------------:|-----------:|---------------------------:|-------------------------:|----------------------:|---------------------:|---------------------:|-------------------:|-----------------------:|--------------------------:|---------------------:|----------------------:|-----------------------:|------------------:|:-------------------------------------|--------:|-----------------------:|:-------------|-----------:|---------:|------------------:|-------------:|------------------------:|----------------:|-----------------:|-------------:|----------------:|-----------------:|--------------------:|-------------------:|--------------:|--------------:|---------------:|-----------------:|-------------------:|--------------:|------------:|---------:|:---------------|:-------------------------|:----------------------|-------------:|:------------------|:-----------------------------------------|---------:|------------------:|-----------------:|--------------------:|--------------------:|-------------------:|------------------:|---------------:|----------------:|--------------------:|-----------------:|------------:|:-----------------------|:---------------------------------|----------:|:----------|:-------------|:--------------|-----------------:|
## | 3663 | nan                  |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |          0 |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | efd13f52-219f-44ca-8c7a-18d1aa5829e7 |    3672 |                    nan | nan          |       9609 |       92 |               nan |          nan |                     nan |             nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |           nan |            nan |              nan |                nan |           nan |         nan |        2 | Regular Play   | nan                      | nan                   |          185 | Barcelona         | ['fbb9e4ec-27b5-47d0-af0d-f3cdef18cbd7'] |        5 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | nan                    | nan                              |       nan | Barcelona | 00:47:05.183 | Half End      |              nan |
## | 3664 | nan                  |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |          0 |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | fbb9e4ec-27b5-47d0-af0d-f3cdef18cbd7 |    3673 |                    nan | nan          |       9609 |       92 |               nan |          nan |                     nan |             nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |           nan |            nan |              nan |                nan |           nan |         nan |        2 | Regular Play   | nan                      | nan                   |          185 | Barcelona         | ['efd13f52-219f-44ca-8c7a-18d1aa5829e7'] |        5 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | nan                    | nan                              |       nan | Eibar     | 00:47:05.183 | Half End      |              nan |
## | 3665 | nan                  |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |          0 |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | 42bb2ab4-6a90-47c6-8c21-f0ef00301612 |    2602 |                    nan | nan          |       9609 |       62 |               nan |          nan |                     nan |             nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |           nan |            nan |              nan |                nan |           nan |         nan |        2 | From Kick Off  | Andrés Iniesta Luján     | Left Center Midfield  |          126 | Eibar             | nan                                      |       44 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | Tactical               | Sergi Roberto Carnicer           |       nan | Barcelona | 00:17:44.157 | Substitution  |              nan |
## | 3666 | nan                  |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |          0 |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | 7dbbf40f-3042-4f82-80ba-8d33299a5df0 |    2674 |                    nan | nan          |       9609 |       64 |               nan |          nan |                     nan |             nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |           nan |            nan |              nan |                nan |           nan |         nan |        2 | From Throw In  | Ander Capa Rodríguez     | Right Wing            |          129 | Barcelona         | nan                                      |       27 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | Tactical               | Rubén Peña Jiménez               |       nan | Eibar     | 00:19:27.420 | Substitution  |              nan |
## | 3667 | nan                  |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |          0 |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | 9eb5f156-f572-4e63-bb81-8b7908b42a90 |    2675 |                    nan | nan          |       9609 |       64 |               nan |          nan |                     nan |             nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |           nan |            nan |              nan |                nan |           nan |         nan |        2 | From Throw In  | Sergio Busquets i Burgos | Center Midfield       |          129 | Barcelona         | nan                                      |       38 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | Tactical               | Ivan Rakitić                     |       nan | Barcelona | 00:19:38.805 | Substitution  |              nan |
## | 3668 | nan                  |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |          0 |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | d2179ee6-e381-4c3a-8c20-d858497b329e |    2906 |                    nan | nan          |       9609 |       71 |               nan |          nan |                     nan |             nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |           nan |            nan |              nan |                nan |           nan |         nan |        2 | Regular Play   | Sergio Enrich Ametller   | Center Forward        |          144 | Eibar             | nan                                      |       25 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | Tactical               | Charles Días Barbosa de Oliveira |       nan | Eibar     | 00:26:25.986 | Substitution  |              nan |
## | 3669 | nan                  |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |          0 |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | b30c240b-74ee-4a8e-a8f8-6e335605024b |    2993 |                    nan | nan          |       9609 |       73 |               nan |          nan |                     nan |             nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |           nan |            nan |              nan |                nan |           nan |         nan |        2 | From Throw In  | Gerard Deulofeu Lázaro   | Right Wing            |          149 | Eibar             | nan                                      |       44 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | Tactical               | Aleix Vidal Parreu               |       nan | Barcelona | 00:28:44.106 | Substitution  |              nan |
## | 3670 | nan                  |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |          0 |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | bb629ec7-c862-4ae7-a4b1-177cdeacf14a |    3078 |                    nan | nan          |       9609 |       76 |               nan |          nan |                     nan |             nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |           nan |            nan |              nan |                nan |           nan |         nan |        2 | Regular Play   | Daniel García Carrillo   | Right Center Midfield |          152 | Barcelona         | nan                                      |       43 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | Tactical               | Christian Rivera Hernández       |       nan | Eibar     | 00:31:43.301 | Substitution  |              nan |
## | 3671 | nan                  |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |          0 |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | 3144c78c-751b-4f18-a143-6a6f8bb45856 |    2902 |                    nan | [15.0, 36.0] |       9609 |       70 |               nan |          nan |                     nan |             nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |           nan |            nan |              nan |                nan |           nan |         nan |        2 | From Counter   | Alejandro Gálvez Jimena  | Left Center Back      |          143 | Barcelona         | nan                                      |       41 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | nan                    | nan                              |       nan | Eibar     | 00:25:41.224 | Error         |              nan |
## | 3672 | Yellow Card          |                    nan |                              nan |                nan |                  nan |                    nan |            nan |               nan |               nan |            nan |         nan |          0 |                        nan |                      nan |                   nan |                  nan |                  nan |                nan |                    nan |                       nan |                  nan |                   nan |                    nan |               nan | d1e5f981-ed16-46bd-ae46-a164f2ab4eed |    3218 |                    nan | nan          |       9609 |       79 |               nan |          nan |                     nan |             nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |           nan |            nan |              nan |                nan |           nan |         nan |        2 | Regular Play   | Alejandro Gálvez Jimena  | Left Center Back      |          158 | Barcelona         | nan                                      |       52 |               nan |              nan |                 nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | nan                    | nan                              |       nan | Eibar     | 00:34:52.199 | Bad Behaviour |              nan |
```

We have extracted the event data from the `Barcelona` vs. `Eibar` match, which `Barcelona` won *6-1* at home (*Camp Nou*). Interesting!

Let us now look into the column names of `events`, such that we can only extract appropriate columns for visualizing and analyzing our convex hulls.


```python
print(events.columns)
```

```
## Index(['bad_behaviour_card', 'ball_receipt_outcome',
##        'ball_recovery_recovery_failure', 'block_deflection',
##        'carry_end_location', 'clearance_aerial_won', 'counterpress',
##        'dribble_outcome', 'dribble_overrun', 'duel_outcome', 'duel_type',
##        'duration', 'foul_committed_advantage', 'foul_committed_penalty',
##        'foul_committed_type', 'foul_won_advantage', 'foul_won_defensive',
##        'foul_won_penalty', 'goalkeeper_body_part', 'goalkeeper_end_location',
##        'goalkeeper_outcome', 'goalkeeper_position', 'goalkeeper_technique',
##        'goalkeeper_type', 'id', 'index', 'interception_outcome', 'location',
##        'match_id', 'minute', 'pass_aerial_won', 'pass_angle',
##        'pass_assisted_shot_id', 'pass_backheel', 'pass_body_part',
##        'pass_cross', 'pass_cut_back', 'pass_deflected', 'pass_end_location',
##        'pass_goal_assist', 'pass_height', 'pass_length', 'pass_outcome',
##        'pass_recipient', 'pass_shot_assist', 'pass_switch', 'pass_type',
##        'period', 'play_pattern', 'player', 'position', 'possession',
##        'possession_team', 'related_events', 'second', 'shot_aerial_won',
##        'shot_body_part', 'shot_end_location', 'shot_freeze_frame',
##        'shot_key_pass_id', 'shot_one_on_one', 'shot_outcome', 'shot_redirect',
##        'shot_statsbomb_xg', 'shot_technique', 'shot_type',
##        'substitution_outcome', 'substitution_replacement', 'tactics', 'team',
##        'timestamp', 'type', 'under_pressure'],
##       dtype='object')
```

Now let us pick the important columns from the `events` dataset:


```python
events_hull = events[['team', 'location', 'type', 'player']]
print(events_hull.head(10).to_markdown())
```

```
## |    | team      | location     | type        | player                         |
## |---:|:----------|:-------------|:------------|:-------------------------------|
## |  0 | Barcelona | nan          | Starting XI | nan                            |
## |  1 | Eibar     | nan          | Starting XI | nan                            |
## |  2 | Eibar     | nan          | Half Start  | nan                            |
## |  3 | Barcelona | nan          | Half Start  | nan                            |
## |  4 | Barcelona | nan          | Half Start  | nan                            |
## |  5 | Eibar     | nan          | Half Start  | nan                            |
## |  6 | Barcelona | [60.0, 40.0] | Pass        | Lionel Andrés Messi Cuccittini |
## |  7 | Barcelona | [46.0, 36.0] | Pass        | Sergio Busquets i Burgos       |
## |  8 | Barcelona | [56.0, 28.0] | Pass        | Andrés Iniesta Luján           |
## |  9 | Barcelona | [31.0, 27.0] | Pass        | Javier Alejandro Mascherano    |
```

Seems like we only need four columns for now. As we are only focusing on pass and shot events, we will first filter the dataset by setting `type` to `Pass` or `Shot`.


```python
events_hull = events_hull[(events_hull['type'] == 'Pass') | (events_hull['type'] == 'Shot')].reset_index()
print(events_hull.head(10).to_markdown())
```

```
## |    |   index | team      | location     | type   | player                         |
## |---:|--------:|:----------|:-------------|:-------|:-------------------------------|
## |  0 |       6 | Barcelona | [60.0, 40.0] | Pass   | Lionel Andrés Messi Cuccittini |
## |  1 |       7 | Barcelona | [46.0, 36.0] | Pass   | Sergio Busquets i Burgos       |
## |  2 |       8 | Barcelona | [56.0, 28.0] | Pass   | Andrés Iniesta Luján           |
## |  3 |       9 | Barcelona | [31.0, 27.0] | Pass   | Javier Alejandro Mascherano    |
## |  4 |      10 | Barcelona | [32.0, 60.0] | Pass   | Gerard Piqué Bernabéu          |
## |  5 |      11 | Barcelona | [37.0, 74.0] | Pass   | Nélson Cabral Semedo           |
## |  6 |      12 | Barcelona | [5.0, 60.0]  | Pass   | Gerard Piqué Bernabéu          |
## |  7 |      13 | Barcelona | [3.0, 37.0]  | Pass   | Marc-André ter Stegen          |
## |  8 |      14 | Barcelona | [9.0, 20.0]  | Pass   | Javier Alejandro Mascherano    |
## |  9 |      15 | Barcelona | [3.0, 39.0]  | Pass   | Marc-André ter Stegen          |
```

Then, we will split the `location` column into `location_x` and `location_y` columns:


```python
Loc = events_hull['location']
Loc = pd.DataFrame(Loc.to_list(), columns=['location_x', 'location_y'])
events_hull['location_x'] = Loc['location_x']
events_hull['location_y'] = Loc['location_y']
print(events_hull.head(10).to_markdown())
```

```
## |    |   index | team      | location     | type   | player                         |   location_x |   location_y |
## |---:|--------:|:----------|:-------------|:-------|:-------------------------------|-------------:|-------------:|
## |  0 |       6 | Barcelona | [60.0, 40.0] | Pass   | Lionel Andrés Messi Cuccittini |           60 |           40 |
## |  1 |       7 | Barcelona | [46.0, 36.0] | Pass   | Sergio Busquets i Burgos       |           46 |           36 |
## |  2 |       8 | Barcelona | [56.0, 28.0] | Pass   | Andrés Iniesta Luján           |           56 |           28 |
## |  3 |       9 | Barcelona | [31.0, 27.0] | Pass   | Javier Alejandro Mascherano    |           31 |           27 |
## |  4 |      10 | Barcelona | [32.0, 60.0] | Pass   | Gerard Piqué Bernabéu          |           32 |           60 |
## |  5 |      11 | Barcelona | [37.0, 74.0] | Pass   | Nélson Cabral Semedo           |           37 |           74 |
## |  6 |      12 | Barcelona | [5.0, 60.0]  | Pass   | Gerard Piqué Bernabéu          |            5 |           60 |
## |  7 |      13 | Barcelona | [3.0, 37.0]  | Pass   | Marc-André ter Stegen          |            3 |           37 |
## |  8 |      14 | Barcelona | [9.0, 20.0]  | Pass   | Javier Alejandro Mascherano    |            9 |           20 |
## |  9 |      15 | Barcelona | [3.0, 39.0]  | Pass   | Marc-André ter Stegen          |            3 |           39 |
```

we can discard the `location` column:


```python
events_hull = events_hull[['team', 'type', 'player', 'location_x', 'location_y']]
```

We will next split the data into two datasets, one for `Barcelona` and the other for `Eibar`:


```python
events_hull_Barca = events_hull[events_hull['team'] == 'Barcelona'].reset_index()
events_hull_Eibar = events_hull[events_hull['team'] == 'Eibar'].reset_index()
```

Let us look into the first 10 rows of these datasets:


```python
print(events_hull_Barca.head(10).to_markdown())
```

```
## |    |   index | team      | type   | player                         |   location_x |   location_y |
## |---:|--------:|:----------|:-------|:-------------------------------|-------------:|-------------:|
## |  0 |       0 | Barcelona | Pass   | Lionel Andrés Messi Cuccittini |           60 |           40 |
## |  1 |       1 | Barcelona | Pass   | Sergio Busquets i Burgos       |           46 |           36 |
## |  2 |       2 | Barcelona | Pass   | Andrés Iniesta Luján           |           56 |           28 |
## |  3 |       3 | Barcelona | Pass   | Javier Alejandro Mascherano    |           31 |           27 |
## |  4 |       4 | Barcelona | Pass   | Gerard Piqué Bernabéu          |           32 |           60 |
## |  5 |       5 | Barcelona | Pass   | Nélson Cabral Semedo           |           37 |           74 |
## |  6 |       6 | Barcelona | Pass   | Gerard Piqué Bernabéu          |            5 |           60 |
## |  7 |       7 | Barcelona | Pass   | Marc-André ter Stegen          |            3 |           37 |
## |  8 |       8 | Barcelona | Pass   | Javier Alejandro Mascherano    |            9 |           20 |
## |  9 |       9 | Barcelona | Pass   | Marc-André ter Stegen          |            3 |           39 |
```

```python
print(events_hull_Eibar.head(10).to_markdown())
```

```
## |    |   index | team   | type   | player                            |   location_x |   location_y |
## |---:|--------:|:-------|:-------|:----------------------------------|-------------:|-------------:|
## |  0 |      15 | Eibar  | Pass   | Daniel García Carrillo            |           74 |           30 |
## |  1 |      16 | Eibar  | Pass   | Gonzalo Escalante                 |           68 |           28 |
## |  2 |      17 | Eibar  | Pass   | Joan Jordán Moreno                |           77 |           30 |
## |  3 |      18 | Eibar  | Pass   | Daniel García Carrillo            |           69 |           42 |
## |  4 |      26 | Eibar  | Pass   | Gonzalo Escalante                 |           71 |           13 |
## |  5 |      27 | Eibar  | Pass   | Sergio Enrich Ametller            |           86 |           11 |
## |  6 |      28 | Eibar  | Pass   | Joan Jordán Moreno                |           92 |            6 |
## |  7 |      29 | Eibar  | Pass   | Takashi Inui                      |           80 |           10 |
## |  8 |      30 | Eibar  | Pass   | Daniel García Carrillo            |           68 |           31 |
## |  9 |      31 | Eibar  | Pass   | Paulo André Rodrigues de Oliveira |           48 |           43 |
```

Next, we will list down the name of the players from both the teams:


```python
players_Barca = events_hull_Barca.player.unique()
players_Eibar = events_hull_Eibar.player.unique()
print(players_Barca)
```

```
## ['Lionel Andrés Messi Cuccittini' 'Sergio Busquets i Burgos'
##  'Andrés Iniesta Luján' 'Javier Alejandro Mascherano'
##  'Gerard Piqué Bernabéu' 'Nélson Cabral Semedo' 'Marc-André ter Stegen'
##  'Gerard Deulofeu Lázaro' 'Lucas Digne' 'Denis Suárez Fernández'
##  'José Paulo Bezzera Maciel Júnior' 'Sergi Roberto Carnicer'
##  'Ivan Rakitić' 'Aleix Vidal Parreu']
```

```python
print(players_Eibar)
```

```
## ['Daniel García Carrillo' 'Gonzalo Escalante' 'Joan Jordán Moreno'
##  'Sergio Enrich Ametller' 'Takashi Inui'
##  'Paulo André Rodrigues de Oliveira' 'Anaitz Arbilla Zabala'
##  'Alejandro Gálvez Jimena' 'David Juncà Reñé' 'Marko Dmitrović'
##  'Ander Capa Rodríguez' 'Rubén Peña Jiménez'
##  'Charles Días Barbosa de Oliveira' 'Christian Rivera Hernández']
```

First we will focus on computing and visualizing the convex hull for a particular player. What can be a better way to pay respect 🙌 🙌  to one of the greatest midfielders of all time `'Andrés Iniesta Luján'` than picking him up for the analysis? 

We will now extract the event data for `Iniesta` from `events_hull_Barca`.


```python
events_hull_iniesta = events_hull_Barca[events_hull_Barca['player'] == 'Andrés Iniesta Luján']
print(events_hull_iniesta.to_markdown())
```

```
## |     |   index | team      | type   | player               |   location_x |   location_y |
## |----:|--------:|:----------|:-------|:---------------------|-------------:|-------------:|
## |   2 |       2 | Barcelona | Pass   | Andrés Iniesta Luján |           56 |           28 |
## |  24 |      35 | Barcelona | Pass   | Andrés Iniesta Luján |           53 |           21 |
## |  31 |      42 | Barcelona | Pass   | Andrés Iniesta Luján |           72 |           19 |
## |  33 |      44 | Barcelona | Pass   | Andrés Iniesta Luján |           84 |           19 |
## |  38 |      53 | Barcelona | Pass   | Andrés Iniesta Luján |           52 |           28 |
## |  47 |      66 | Barcelona | Pass   | Andrés Iniesta Luján |           52 |           31 |
## |  49 |      74 | Barcelona | Pass   | Andrés Iniesta Luján |           64 |           29 |
## |  67 |     103 | Barcelona | Pass   | Andrés Iniesta Luján |           66 |            8 |
## |  90 |     154 | Barcelona | Pass   | Andrés Iniesta Luján |           54 |           28 |
## |  99 |     167 | Barcelona | Pass   | Andrés Iniesta Luján |           70 |            4 |
## | 106 |     174 | Barcelona | Pass   | Andrés Iniesta Luján |           56 |           24 |
## | 110 |     178 | Barcelona | Pass   | Andrés Iniesta Luján |           79 |            2 |
## | 120 |     189 | Barcelona | Pass   | Andrés Iniesta Luján |           40 |           36 |
## | 128 |     199 | Barcelona | Pass   | Andrés Iniesta Luján |           87 |            7 |
## | 130 |     201 | Barcelona | Pass   | Andrés Iniesta Luján |           80 |            8 |
## | 139 |     225 | Barcelona | Pass   | Andrés Iniesta Luján |           37 |           23 |
## | 140 |     240 | Barcelona | Pass   | Andrés Iniesta Luján |           48 |           23 |
## | 145 |     245 | Barcelona | Pass   | Andrés Iniesta Luján |           97 |           50 |
## | 151 |     259 | Barcelona | Pass   | Andrés Iniesta Luján |           42 |           13 |
## | 153 |     261 | Barcelona | Pass   | Andrés Iniesta Luján |           66 |           26 |
## | 157 |     266 | Barcelona | Pass   | Andrés Iniesta Luján |           44 |           25 |
## | 178 |     301 | Barcelona | Pass   | Andrés Iniesta Luján |           90 |           23 |
## | 182 |     311 | Barcelona | Pass   | Andrés Iniesta Luján |           34 |           28 |
## | 185 |     315 | Barcelona | Pass   | Andrés Iniesta Luján |           60 |           10 |
## | 210 |     375 | Barcelona | Pass   | Andrés Iniesta Luján |           54 |            5 |
## | 212 |     377 | Barcelona | Pass   | Andrés Iniesta Luján |           34 |           14 |
## | 226 |     413 | Barcelona | Pass   | Andrés Iniesta Luján |           39 |           21 |
## | 232 |     419 | Barcelona | Pass   | Andrés Iniesta Luján |           69 |           18 |
## | 248 |     435 | Barcelona | Pass   | Andrés Iniesta Luján |           92 |           12 |
## | 263 |     459 | Barcelona | Pass   | Andrés Iniesta Luján |           64 |            4 |
## | 290 |     499 | Barcelona | Pass   | Andrés Iniesta Luján |           44 |            9 |
## | 304 |     536 | Barcelona | Pass   | Andrés Iniesta Luján |           62 |            6 |
## | 314 |     546 | Barcelona | Pass   | Andrés Iniesta Luján |           71 |           26 |
## | 316 |     548 | Barcelona | Pass   | Andrés Iniesta Luján |           68 |           16 |
## | 331 |     574 | Barcelona | Pass   | Andrés Iniesta Luján |           49 |           20 |
## | 337 |     581 | Barcelona | Pass   | Andrés Iniesta Luján |           50 |           14 |
## | 345 |     589 | Barcelona | Pass   | Andrés Iniesta Luján |           51 |           15 |
## | 391 |     680 | Barcelona | Pass   | Andrés Iniesta Luján |           52 |           25 |
## | 399 |     688 | Barcelona | Pass   | Andrés Iniesta Luján |          101 |           36 |
## | 408 |     699 | Barcelona | Pass   | Andrés Iniesta Luján |           81 |           47 |
## | 410 |     701 | Barcelona | Pass   | Andrés Iniesta Luján |           83 |           49 |
```

Before computing and visualizing the convex hull, it is a good practice to discard the outliers from the datasets. A common method that researchers use is the [*Inter Quartile Range*](https://en.wikipedia.org/wiki/Interquartile_range). We will find the inter quartile ranges for the columns `location_x` and `location_y` from `events_hull_iniesta` and then compute the upper and lower bounds of the data. Any points lying beyond these bounds, i.e any point lying above the lower bound and any point lying below the upper bound, are decided to be *outliers* and are discarded. We use box plots and whisker plots to visualize the interquartile range for the datapoints:  


```python
e_box = pd.DataFrame(data = events_hull_iniesta, columns = ["location_x", "location_y"])
boxplot = sns.boxplot(x = "variable", y ="value", data=pd.melt(e_box), order = ["location_x", "location_y"])
boxplot = sns.stripplot(x = "variable", y = "value", data = pd.melt(e_box), marker="o", color="red", order = ["location_x", "location_y"])
boxplot.axes.set_title("Boxplot for Iniesta's location conditions")
```

```python
plt.show()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-1.png" width="672" />

We will next compute the quartiles, the inter quartile range and the minimum and maximum values:


```python
Q1 = np.percentile(events_hull_iniesta['location_x'], 25, interpolation='midpoint')
Q3 = np.percentile(events_hull_iniesta['location_x'], 75, interpolation='midpoint')
IQR_x = Q3 - Q1

minimum_x = Q1 - 1.5*IQR_x
maximum_x = Q3 + 1.5*IQR_x
Q1, Q3, IQR_x, minimum_x, maximum_x
```

```
## (50.0, 72.0, 22.0, 17.0, 105.0)
```


```python
Q1 = np.percentile(events_hull_iniesta['location_y'], 25, interpolation='midpoint')
Q3 = np.percentile(events_hull_iniesta['location_y'], 75, interpolation='midpoint')
IQR_y = Q3 - Q1

minimum_y = Q1 - 1.5*IQR_y
maximum_y = Q3 + 1.5*IQR_y
Q1, Q3, IQR_y, minimum_y, maximum_y
```

```
## (12.0, 28.0, 16.0, -12.0, 52.0)
```


```python
upper = np.where((events_hull_iniesta['location_x'] >= maximum_x) & (events_hull_iniesta['location_y'] >= maximum_y))
lower = np.where((events_hull_iniesta['location_x'] <= minimum_x) & (events_hull_iniesta['location_y'] <= minimum_y))
```

Finally, we will drop the outliers if present:


```python
events_hull_iniesta.drop(upper[0], inplace = True)
```

```python
events_hull_iniesta.drop(lower[0], inplace = True)
```

Let us again print the top 10 rows of the `events_hull_iniesta` dataset:


```python
events_hull_iniesta = events_hull_iniesta.reset_index()
events_hull_iniesta = events_hull_iniesta[['team', 'type', 'player', 'location_x', 'location_y']]
print(events_hull_iniesta.to_markdown())
```

```
## |    | team      | type   | player               |   location_x |   location_y |
## |---:|:----------|:-------|:---------------------|-------------:|-------------:|
## |  0 | Barcelona | Pass   | Andrés Iniesta Luján |           56 |           28 |
## |  1 | Barcelona | Pass   | Andrés Iniesta Luján |           53 |           21 |
## |  2 | Barcelona | Pass   | Andrés Iniesta Luján |           72 |           19 |
## |  3 | Barcelona | Pass   | Andrés Iniesta Luján |           84 |           19 |
## |  4 | Barcelona | Pass   | Andrés Iniesta Luján |           52 |           28 |
## |  5 | Barcelona | Pass   | Andrés Iniesta Luján |           52 |           31 |
## |  6 | Barcelona | Pass   | Andrés Iniesta Luján |           64 |           29 |
## |  7 | Barcelona | Pass   | Andrés Iniesta Luján |           66 |            8 |
## |  8 | Barcelona | Pass   | Andrés Iniesta Luján |           54 |           28 |
## |  9 | Barcelona | Pass   | Andrés Iniesta Luján |           70 |            4 |
## | 10 | Barcelona | Pass   | Andrés Iniesta Luján |           56 |           24 |
## | 11 | Barcelona | Pass   | Andrés Iniesta Luján |           79 |            2 |
## | 12 | Barcelona | Pass   | Andrés Iniesta Luján |           40 |           36 |
## | 13 | Barcelona | Pass   | Andrés Iniesta Luján |           87 |            7 |
## | 14 | Barcelona | Pass   | Andrés Iniesta Luján |           80 |            8 |
## | 15 | Barcelona | Pass   | Andrés Iniesta Luján |           37 |           23 |
## | 16 | Barcelona | Pass   | Andrés Iniesta Luján |           48 |           23 |
## | 17 | Barcelona | Pass   | Andrés Iniesta Luján |           97 |           50 |
## | 18 | Barcelona | Pass   | Andrés Iniesta Luján |           42 |           13 |
## | 19 | Barcelona | Pass   | Andrés Iniesta Luján |           66 |           26 |
## | 20 | Barcelona | Pass   | Andrés Iniesta Luján |           44 |           25 |
## | 21 | Barcelona | Pass   | Andrés Iniesta Luján |           90 |           23 |
## | 22 | Barcelona | Pass   | Andrés Iniesta Luján |           34 |           28 |
## | 23 | Barcelona | Pass   | Andrés Iniesta Luján |           60 |           10 |
## | 24 | Barcelona | Pass   | Andrés Iniesta Luján |           54 |            5 |
## | 25 | Barcelona | Pass   | Andrés Iniesta Luján |           34 |           14 |
## | 26 | Barcelona | Pass   | Andrés Iniesta Luján |           39 |           21 |
## | 27 | Barcelona | Pass   | Andrés Iniesta Luján |           69 |           18 |
## | 28 | Barcelona | Pass   | Andrés Iniesta Luján |           92 |           12 |
## | 29 | Barcelona | Pass   | Andrés Iniesta Luján |           64 |            4 |
## | 30 | Barcelona | Pass   | Andrés Iniesta Luján |           44 |            9 |
## | 31 | Barcelona | Pass   | Andrés Iniesta Luján |           62 |            6 |
## | 32 | Barcelona | Pass   | Andrés Iniesta Luján |           71 |           26 |
## | 33 | Barcelona | Pass   | Andrés Iniesta Luján |           68 |           16 |
## | 34 | Barcelona | Pass   | Andrés Iniesta Luján |           49 |           20 |
## | 35 | Barcelona | Pass   | Andrés Iniesta Luján |           50 |           14 |
## | 36 | Barcelona | Pass   | Andrés Iniesta Luján |           51 |           15 |
## | 37 | Barcelona | Pass   | Andrés Iniesta Luján |           52 |           25 |
## | 38 | Barcelona | Pass   | Andrés Iniesta Luján |          101 |           36 |
## | 39 | Barcelona | Pass   | Andrés Iniesta Luján |           81 |           47 |
## | 40 | Barcelona | Pass   | Andrés Iniesta Luján |           83 |           49 |
```

First we collect all the points from the two columns as a 2-D matrix. This comes in aid while drawing the convex hull.


```python
points_hull = events_hull_iniesta[['location_x', 'location_y']].values
print(points_hull)
```

```
## [[ 56.  28.]
##  [ 53.  21.]
##  [ 72.  19.]
##  [ 84.  19.]
##  [ 52.  28.]
##  [ 52.  31.]
##  [ 64.  29.]
##  [ 66.   8.]
##  [ 54.  28.]
##  [ 70.   4.]
##  [ 56.  24.]
##  [ 79.   2.]
##  [ 40.  36.]
##  [ 87.   7.]
##  [ 80.   8.]
##  [ 37.  23.]
##  [ 48.  23.]
##  [ 97.  50.]
##  [ 42.  13.]
##  [ 66.  26.]
##  [ 44.  25.]
##  [ 90.  23.]
##  [ 34.  28.]
##  [ 60.  10.]
##  [ 54.   5.]
##  [ 34.  14.]
##  [ 39.  21.]
##  [ 69.  18.]
##  [ 92.  12.]
##  [ 64.   4.]
##  [ 44.   9.]
##  [ 62.   6.]
##  [ 71.  26.]
##  [ 68.  16.]
##  [ 49.  20.]
##  [ 50.  14.]
##  [ 51.  15.]
##  [ 52.  25.]
##  [101.  36.]
##  [ 81.  47.]
##  [ 83.  49.]]
```

Now, let us use the `ConvexHull()` function from `scipy.spatial`:


```python
convex_hull_iniesta = ConvexHull(events_hull_iniesta[['location_x', 'location_y']])
```

This *convex hull* is represented by the *vertices*, i.e the coordinate points that make the vertices of the convex hull and the *simplices*, i.e the stratight line in case of a 2-D plane that connects the *vertices* of the the *convex hull*. The `vertices` attribute consists of the indices of the points in `points_hull` that make up the convex hull, and the `simplices` attribute too consists of the indices of the points in `points_hull`. The `simplices` are a list of 1-D simplices of a particular length, representing line segments in 2-D. Let us print the indices:


```python
print(convex_hull_iniesta.vertices)
```

```
## [22 25 30 24 11 13 28 38 17 40 12]
```

```python
print(convex_hull_iniesta.simplices)
```

```
## [[25 22]
##  [28 38]
##  [12 22]
##  [12 40]
##  [17 38]
##  [17 40]
##  [24 11]
##  [13 11]
##  [13 28]
##  [30 25]
##  [30 24]]
```

Now we have collected all the useful information and will visualize the convex hull on a football pitch:


```python
pitch = Pitch(pitch_color='black', line_color='white', goal_type='box', 
              constrained_layout=True, tight_layout=False)
fig, ax = pitch.draw()

plt.scatter(events_hull_iniesta.location_x, events_hull_iniesta.location_y, color='red')
```

```python
for i in convex_hull_iniesta.simplices:
    plt.plot(points_hull[i, 0], points_hull[i, 1], 'white')
    plt.fill(points_hull[convex_hull_iniesta.vertices, 0], points_hull[convex_hull_iniesta.vertices, 1], c='white', alpha=0.1)
```

```python
plt.title("Convex Hull for Iniesta's field coverage against Eibar")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-3.png" width="672" />

So, we see that `Iniesta` mostly covered the left side of `Barcelona`'s attack on the field. That speaks a lot! Now let us go ahead and compute and visualize the convex hull for all the players in a team


```python
ccodes = ['#FAEBD7', '#66CDAA', '#E3CF57', '#8A2BE2', '#EE3B3B', '#66CD00', '#DC143C', '#FFB90F', '#A9A9A9', '#B23AEE', '#CD1076',
         '#8B6914', '#BFEFFF', '#EED2EE', '#C6E2FF', '#C67171']
         
fig, axes = plt.subplots(6, 3, figsize = (20, 25))
axes = axes.ravel()

for idx, p in enumerate(players_Barca):
    pitch = Pitch(pitch_color='black', line_color='white', goal_type='box', 
              constrained_layout=True, tight_layout=False)
    pitch.draw(axes[idx])
    xmin, xmax, ymin, ymax = pitch.extent
    
    axes[idx].xaxis.set_ticks([xmin, xmax])
    axes[idx].yaxis.set_ticks([ymin, ymax])
    axes[idx].tick_params(labelsize=15)
    axes[idx].set_title(p, fontsize=20, pad=8)
    
    Eh = events_hull_Barca[events_hull_Barca['player'] == p].reset_index()
    
    Q1 = np.percentile(Eh['location_x'], 25, interpolation='midpoint')
    Q3 = np.percentile(Eh['location_x'], 75, interpolation='midpoint')
    IQR_x = Q3 - Q1

    minimum_x = Q1 - 1.5*IQR_x
    maximum_x = Q3 + 1.5*IQR_x
    
    Q1 = np.percentile(Eh['location_y'], 25, interpolation='midpoint')
    Q3 = np.percentile(Eh['location_y'], 75, interpolation='midpoint')
    IQR_y = Q3 - Q1

    minimum_y = Q1 - 1.5*IQR_y
    maximum_y = Q3 + 1.5*IQR_y
    
    upper = np.where((Eh['location_x'] >= maximum_x) & (Eh['location_y'] >= maximum_y))
    lower = np.where((Eh['location_x'] <= minimum_x) & (Eh['location_y'] <= minimum_y))
    
    Eh.drop(upper[0], inplace = True)
    Eh.drop(lower[0], inplace = True)
    
    points_Barca = Eh[['location_x', 'location_y']].values
    convex_hull_Barca = ConvexHull(Eh[['location_x', 'location_y']])
    
    axes[idx].scatter(Eh.location_x, Eh.location_y, color='red')
    
    for i in convex_hull_Barca.simplices:
            axes[idx].plot(points_Barca[i, 0], points_Barca[i, 1], 'white')
            axes[idx].fill(points_Barca[convex_hull_Barca.vertices, 0], points_Barca[convex_hull_Barca.vertices, 1], c=ccodes[idx], alpha=0.1)
            
```

```python
title = fig.suptitle("Convex Hulls for Barcelona players' field coverage vs Eibar [La Liga 2017-18]", fontsize=33)

for j in range(len(players_Barca)-18, 0):
    axes[j].remove()
plt.show()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-5.png" width="1920" />

Now, we will compute for `Eibar`'s team players:


```python
ccodes = ['#FAEBD7', '#66CDAA', '#E3CF57', '#8A2BE2', '#EE3B3B', '#66CD00', '#DC143C', '#FFB90F', '#A9A9A9', '#B23AEE', '#CD1076',
         '#8B6914', '#BFEFFF', '#EED2EE', '#C6E2FF', '#C67171']
         
fig, axes = plt.subplots(6, 3, figsize = (20, 25))
axes = axes.ravel()

for idx, p in enumerate(players_Eibar):
    pitch = Pitch(pitch_color='black', line_color='white', goal_type='box', 
              constrained_layout=True, tight_layout=False)
    pitch.draw(axes[idx])
    xmin, xmax, ymin, ymax = pitch.extent
    
    axes[idx].xaxis.set_ticks([xmin, xmax])
    axes[idx].yaxis.set_ticks([ymin, ymax])
    axes[idx].tick_params(labelsize=15)
    axes[idx].set_title(p, fontsize=20, pad=8)
    
    Eh = events_hull_Eibar[events_hull_Eibar['player'] == p].reset_index()
    Eh['location_x'] = 120 - Eh['location_x']
    
    Q1 = np.percentile(Eh['location_x'], 25, interpolation='midpoint')
    Q3 = np.percentile(Eh['location_x'], 75, interpolation='midpoint')
    IQR_x = Q3 - Q1

    minimum_x = Q1 - 1.5*IQR_x
    maximum_x = Q3 + 1.5*IQR_x
    
    Q1 = np.percentile(Eh['location_y'], 25, interpolation='midpoint')
    Q3 = np.percentile(Eh['location_y'], 75, interpolation='midpoint')
    IQR_y = Q3 - Q1

    minimum_y = Q1 - 1.5*IQR_y
    maximum_y = Q3 + 1.5*IQR_y
    
    upper = np.where((Eh['location_x'] >= maximum_x) & (Eh['location_y'] >= maximum_y))
    lower = np.where((Eh['location_x'] <= minimum_x) & (Eh['location_y'] <= minimum_y))
    
    Eh.drop(upper[0], inplace = True)
    Eh.drop(lower[0], inplace = True)
    
    points_Eibar = Eh[['location_x', 'location_y']].values
    convex_hull_Eibar = ConvexHull(Eh[['location_x', 'location_y']])
    
    axes[idx].scatter(Eh.location_x, Eh.location_y, color='red')
    
    for i in convex_hull_Eibar.simplices:
            axes[idx].plot(points_Eibar[i, 0], points_Eibar[i, 1], 'white')
            axes[idx].fill(points_Eibar[convex_hull_Eibar.vertices, 0], points_Eibar[convex_hull_Eibar.vertices, 1], c=ccodes[idx], alpha=0.1)
            
```

```python
title = fig.suptitle("Convex Hulls for Eiber players' field coverage vs Barcelona [La Liga 2017-18]", fontsize=33)

for j in range(len(players_Eibar)-18, 0):
    axes[j].remove()
plt.show()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-7.png" width="1920" />

So, we have been able to compute and visualize the convex hulls for players from a particular game. Next, we will try to understand how to get tracking data from a particular game using `statsbomb` api. We need tracking data to compute *Delaunay triangulations* and *Voronoi diagrams*.


**This post is still under construction**
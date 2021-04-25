---
title: Introduction
author: Indranil Ghosh
date: '2021-04-25'
slug: introduction
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2021-04-25T01:51:34+05:30'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

This is a dummy website to try using simple functionalities. A simple R code to import all the packages:



Install the Following Python package given below:


```python
from statsbombpy import sb
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
```

Now show the competition data

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
## | 15 |               37 |          42 | England                  | FA Women's Super League | female               | 2019/2020     | 2020-10-20T18:35:33.568528 | 2020-10-20T18:35:33.568528 |
## | 16 |               37 |           4 | England                  | FA Women's Super League | female               | 2018/2019     | 2021-04-09T20:39:24.201269 | 2021-04-09T20:39:24.201269 |
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

We will now look into the matches available:

```python
mat = sb.matches(competition_id=11, season_id=42)
```

```
## credentials were not supplied. open data access only
```

```python
print(mat.to_markdown())
```

```
## |    |   match_id | match_date   | kick_off     | competition     | season    | home_team        | away_team        |   home_score |   away_score | match_status   | match_status_360   | last_updated               | last_updated_360   |   match_week | competition_stage   | stadium                         | referee           | data_version   |   shot_fidelity_version |   xy_fidelity_version |
## |---:|-----------:|:-------------|:-------------|:----------------|:----------|:-----------------|:-----------------|-------------:|-------------:|:---------------|:-------------------|:---------------------------|:-------------------|-------------:|:--------------------|:--------------------------------|:------------------|:---------------|------------------------:|----------------------:|
## |  0 |     303421 | 2020-07-19   | 17:00:00.000 | Spain - La Liga | 2019/2020 | Deportivo Alavés | Barcelona        |            0 |            5 | available      | unscheduled        | 2020-07-29T05:00           |                    |           38 | Regular Season      | Estadio de Mendizorroza         | J. Martínez       | 1.1.0          |                       2 |                     2 |
## |  1 |     303493 | 2020-06-23   | 22:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Athletic Bilbao  |            1 |            0 | available      | unscheduled        | 2020-07-29T05:00           |                    |           31 | Regular Season      | Camp Nou                        | Jesús Gil         | 1.1.0          |                       2 |                     2 |
## |  2 |     303516 | 2020-06-27   | 17:00:00.000 | Spain - La Liga | 2019/2020 | Celta Vigo       | Barcelona        |            2 |            2 | available      | unscheduled        | 2020-07-29T05:00           |                    |           32 | Regular Season      | Abanca-Balaídos                 | G. Cuadra         | 1.1.0          |                       2 |                     2 |
## |  3 |     303680 | 2020-07-11   | 19:30:00.000 | Spain - La Liga | 2019/2020 | Real Valladolid  | Barcelona        |            0 |            1 | available      | unscheduled        | 2020-12-18T12:10:38.985394 |                    |           36 | Regular Season      | Estadio Municipal José Zorrilla | Antonio Mateu     | 1.1.0          |                       2 |                     2 |
## |  4 |     303532 | 2020-06-16   | 22:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Leganés          |            2 |            0 | available      | unscheduled        | 2020-07-29T05:00           |                    |           29 | Regular Season      | Camp Nou                        | J. Martínez       | 1.1.0          |                       2 |                     2 |
## |  5 |     303400 | 2020-01-25   | 16:00:00.000 | Spain - La Liga | 2019/2020 | Valencia         | Barcelona        |            2 |            0 | available      | unscheduled        | 2020-07-29T05:00           |                    |           21 | Regular Season      | Estadio de Mestalla             | Jesús Gil         | 1.1.0          |                       2 |                     2 |
## |  6 |     303634 | 2020-07-16   | 21:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Osasuna          |            1 |            2 | available      | unscheduled        | 2020-09-18T13:16:12.825671 |                    |           37 | Regular Season      | Camp Nou                        | J. Sánchez        | 1.1.0          |                       2 |                     2 |
## |  7 |     303479 | 2020-03-07   | 18:30:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Real Sociedad    |            1 |            0 | available      | unscheduled        | 2020-07-29T05:00           |                    |           27 | Regular Season      | Camp Nou                        | J. Martínez       | 1.1.0          |                       2 |                     2 |
## |  8 |     303615 | 2020-07-08   | 22:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Espanyol         |            1 |            0 | available      | unscheduled        | 2020-09-11T23:12:41.238499 |                    |           35 | Regular Season      | Camp Nou                        | J. Munuera        | 1.1.0          |                       2 |                     2 |
## |  9 |     303696 | 2020-06-30   | 22:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Atlético Madrid  |            2 |            2 | available      | unscheduled        | 2020-07-29T05:00           |                    |           33 | Regular Season      | Camp Nou                        | A. Hernández      | 1.1.0          |                       2 |                     2 |
## | 10 |     303664 | 2019-12-14   | 16:00:00.000 | Spain - La Liga | 2019/2020 | Real Sociedad    | Barcelona        |            2 |            2 | available      | unscheduled        | 2020-07-29T05:00           |                    |           17 | Regular Season      | Reale Arena                     | J. Alberola Rojas | 1.1.0          |                       2 |                     2 |
## | 11 |     303596 | 2019-12-18   | 20:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Real Madrid      |            0 |            0 | available      | unscheduled        | 2020-07-29T05:00           |                    |           10 | Regular Season      | Camp Nou                        | A. Hernández      | 1.1.0          |                       2 |                     2 |
## | 12 |     303487 | 2019-11-09   | 21:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Celta Vigo       |            4 |            1 | available      | unscheduled        | 2020-07-29T05:00           |                    |           13 | Regular Season      | Camp Nou                        | G. Cuadra         | 1.1.0          |                       2 |                     2 |
## | 13 |     303600 | 2019-10-29   | 21:15:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Real Valladolid  |            5 |            1 | available      | unscheduled        | 2020-07-29T05:00           |                    |           11 | Regular Season      | Camp Nou                        | J. Alberola Rojas | 1.1.0          |                       2 |                     2 |
## | 14 |     303548 | 2020-06-13   | 22:00:00.000 | Spain - La Liga | 2019/2020 | Mallorca         | Barcelona        |            0 |            4 | available      | unscheduled        | 2020-07-29T05:00           |                    |           28 | Regular Season      | Iberostar Estadi                | C. Del Cerro      | 1.1.0          |                       2 |                     2 |
## | 15 |     303473 | 2019-10-06   | 21:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Sevilla          |            4 |            0 | available      | unscheduled        | 2020-07-29T05:00           |                    |            8 | Regular Season      | Camp Nou                        | Antonio Mateu     | 1.1.0          |                       2 |                     2 |
## | 16 |     303610 | 2020-01-19   | 21:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Granada          |            1 |            0 | available      | unscheduled        | 2020-07-29T05:00           |                    |           20 | Regular Season      | Camp Nou                        | V. Pizarro        | 1.1.0          |                       2 |                     2 |
## | 17 |     303652 | 2020-01-04   | 21:00:00.000 | Spain - La Liga | 2019/2020 | Espanyol         | Barcelona        |            2 |            2 | available      | unscheduled        | 2020-07-29T05:00           |                    |           19 | Regular Season      | RCDE Stadium                    | C. Del Cerro      | 1.1.0          |                       2 |                     2 |
## | 18 |     303430 | 2019-09-24   | 21:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Villarreal       |            2 |            1 | available      | unscheduled        | 2020-07-29T05:00           |                    |            6 | Regular Season      | Camp Nou                        | R. De Burgos      | 1.1.0          |                       2 |                     2 |
## | 19 |     303674 | 2020-06-19   | 22:00:00.000 | Spain - La Liga | 2019/2020 | Sevilla          | Barcelona        |            0 |            0 | available      | unscheduled        | 2020-07-29T05:00           |                    |           30 | Regular Season      | Estadio Ramón Sánchez Pizjuán   | José González     | 1.1.0          |                       2 |                     2 |
## | 20 |     303470 | 2020-03-01   | 21:00:00.000 | Spain - La Liga | 2019/2020 | Real Madrid      | Barcelona        |            2 |            0 | available      | unscheduled        | 2020-07-29T05:00           |                    |           26 | Regular Season      | Estadio Santiago Bernabéu       | Antonio Mateu     | 1.1.0          |                       2 |                     2 |
## | 21 |     303700 | 2019-10-19   | 13:00:00.000 | Spain - La Liga | 2019/2020 | Eibar            | Barcelona        |            0 |            3 | available      | unscheduled        | 2020-07-29T05:00           |                    |            9 | Regular Season      | Estadio Municipal de Ipurúa     | M. Melero         | 1.1.0          |                       2 |                     2 |
## | 22 |     303707 | 2020-02-09   | 21:00:00.000 | Spain - La Liga | 2019/2020 | Real Betis       | Barcelona        |            2 |            3 | available      | unscheduled        | 2020-07-29T05:00           |                    |           23 | Regular Season      | Estadio Benito Villamarín       | J. Sánchez        | 1.1.0          |                       2 |                     2 |
## | 23 |     303666 | 2019-09-21   | 21:00:00.000 | Spain - La Liga | 2019/2020 | Granada          | Barcelona        |            2 |            0 | available      | unscheduled        | 2020-07-29T05:00           |                    |            5 | Regular Season      | Estadio Nuevo Los Cármenes      | G. Cuadra         | 1.1.0          |                       2 |                     2 |
## | 24 |     303725 | 2020-07-05   | 22:00:00.000 | Spain - La Liga | 2019/2020 | Villarreal       | Barcelona        |            1 |            4 | available      | unscheduled        | 2020-07-29T05:00           |                    |           34 | Regular Season      | Estadio de la Cerámica          | C. Del Cerro      | 1.1.0          |                       2 |                     2 |
## | 25 |     303504 | 2019-11-02   | 16:00:00.000 | Spain - La Liga | 2019/2020 | Levante          | Barcelona        |            3 |            1 | available      | unscheduled        | 2020-07-29T05:00           |                    |           12 | Regular Season      | Estadio Ciudad de Valencia      | A. Hernández      | 1.1.0          |                       2 |                     2 |
## | 26 |     303715 | 2019-11-23   | 13:00:00.000 | Spain - La Liga | 2019/2020 | Leganés          | Barcelona        |            1 |            2 | available      | unscheduled        | 2020-07-29T05:00           |                    |           14 | Regular Season      | Estadio Municipal de Butarque   | S. Jaime          | 1.1.0          |                       2 |                     2 |
## | 27 |     303377 | 2020-02-15   | 16:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Getafe           |            2 |            1 | available      | unscheduled        | 2020-07-29T05:00           |                    |           24 | Regular Season      | Camp Nou                        | G. Cuadra         | 1.1.0          |                       2 |                     2 |
## | 28 |     303524 | 2019-12-01   | 21:00:00.000 | Spain - La Liga | 2019/2020 | Atlético Madrid  | Barcelona        |            0 |            1 | available      | unscheduled        | 2020-07-29T05:00           |                    |           15 | Regular Season      | Estadio Wanda Metropolitano     | Antonio Mateu     | 1.1.0          |                       2 |                     2 |
## | 29 |     303451 | 2019-12-07   | 21:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Mallorca         |            5 |            2 | available      | unscheduled        | 2020-07-29T05:00           |                    |           16 | Regular Season      | Camp Nou                        | J. Munuera        | 1.1.0          |                       2 |                     2 |
## | 30 |     303517 | 2019-12-21   | 16:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Deportivo Alavés |            4 |            1 | available      | unscheduled        | 2020-07-29T05:00           |                    |           18 | Regular Season      | Camp Nou                        | M. Melero         | 1.1.0          |                       2 |                     2 |
## | 31 |     303682 | 2020-02-02   | 21:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Levante          |            2 |            1 | available      | unscheduled        | 2020-07-29T05:00           |                    |           22 | Regular Season      | Camp Nou                        | A. Cordero        | 1.1.0          |                       2 |                     2 |
## | 32 |     303731 | 2020-02-22   | 16:00:00.000 | Spain - La Liga | 2019/2020 | Barcelona        | Eibar            |            5 |            0 | available      | unscheduled        | 2020-07-29T05:00           |                    |           25 | Regular Season      | Camp Nou                        | C. Soto           | 1.1.0          |                       2 |                     2 |
```


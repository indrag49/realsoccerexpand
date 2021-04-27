---
title: Draw a pass map, a shot map and their corresponding heat maps
author: Indranil Ghosh
date: '2021-04-26'
slug: pass-map-shot-map-and-heat-map
categories: ["Python", "visualization"]
tags: ["football pitch", "pass map", "shot map", "heat map", "seaborn", "statsbomb api"]
subtitle: ''
summary: ''
authors: []
lastmod: '2021-04-26T20:27:24+05:30'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

In this post, we will learn how to draw simple pass maps and shot maps and visualize their corresponding heat maps. We will study use the event data from the *Real Madrid vs. Barcelona* match that we discussed about in the [first post](https://realsoccerexpand.netlify.app/post/getting-statsbomb-event-data/). Before that we need to `pip` install the [`seaborn`](https://seaborn.pydata.org/) package which is a Python package built on `matplotlib` and is used for generating informative and appealing statistical graphs for analysis purposes. 


```python
pip install seaborn
```

Let us now import the pertinent packages for this tutorial:


```python
import numpy as np
import pandas as pd
from statsbombpy import sb
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
import seaborn as sns
```

Let us rewrite the code till the point where we were able to extract the event data for the *Madrid vs. Barca* match. The competitions dataset:


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
## | 15 |               37 |          42 | England                  | FA Women's Super League | female               | 2019/2020     | 2021-04-27T06:27:03.599355 | 2021-04-27T06:27:03.599355 |
## | 16 |               37 |           4 | England                  | FA Women's Super League | female               | 2018/2019     | 2021-04-27T08:59:34.340    | 2021-04-09T20:39:24.201269 |
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

The matches dataset:


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

The events dataset:

```python
events = sb.events(match_id=303470)
```

```
## credentials were not supplied. open data access only
```

```python
eh = events.head() # shows the first few rows
print(eh.to_markdown())
```

```
## |    |   bad_behaviour_card |   ball_receipt_outcome |   ball_recovery_recovery_failure |   block_deflection |   block_save_block |   carry_end_location |   clearance_aerial_won |   clearance_body_part |   clearance_head |   clearance_left_foot |   clearance_right_foot |   counterpress |   dribble_no_touch |   dribble_outcome |   dribble_overrun |   duel_outcome |   duel_type |   duration |   foul_committed_advantage |   foul_committed_card |   foul_committed_offensive |   foul_committed_type |   foul_won_advantage |   foul_won_defensive |   goalkeeper_body_part |   goalkeeper_end_location |   goalkeeper_outcome |   goalkeeper_position |   goalkeeper_success_in_play |   goalkeeper_technique |   goalkeeper_type | id                                   |   index |   interception_outcome |   location |   match_id |   minute |   miscontrol_aerial_won |   off_camera |   out |   pass_aerial_won |   pass_angle |   pass_assisted_shot_id |   pass_body_part |   pass_cross |   pass_cut_back |   pass_deflected |   pass_end_location |   pass_goal_assist |   pass_height |   pass_inswinging |   pass_length |   pass_miscommunication |   pass_no_touch |   pass_outcome |   pass_outswinging |   pass_recipient |   pass_shot_assist |   pass_switch |   pass_technique |   pass_through_ball |   pass_type |   period | play_pattern   |   player |   position |   possession | possession_team   | related_events                           |   second |   shot_body_part |   shot_deflected |   shot_end_location |   shot_first_time |   shot_freeze_frame |   shot_key_pass_id |   shot_one_on_one |   shot_outcome |   shot_statsbomb_xg |   shot_technique |   shot_type |   substitution_outcome |   substitution_replacement | tactics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | team        | timestamp    | type        |   under_pressure |
## |---:|---------------------:|-----------------------:|---------------------------------:|-------------------:|-------------------:|---------------------:|-----------------------:|----------------------:|-----------------:|----------------------:|-----------------------:|---------------:|-------------------:|------------------:|------------------:|---------------:|------------:|-----------:|---------------------------:|----------------------:|---------------------------:|----------------------:|---------------------:|---------------------:|-----------------------:|--------------------------:|---------------------:|----------------------:|-----------------------------:|-----------------------:|------------------:|:-------------------------------------|--------:|-----------------------:|-----------:|-----------:|---------:|------------------------:|-------------:|------:|------------------:|-------------:|------------------------:|-----------------:|-------------:|----------------:|-----------------:|--------------------:|-------------------:|--------------:|------------------:|--------------:|------------------------:|----------------:|---------------:|-------------------:|-----------------:|-------------------:|--------------:|-----------------:|--------------------:|------------:|---------:|:---------------|---------:|-----------:|-------------:|:------------------|:-----------------------------------------|---------:|-----------------:|-----------------:|--------------------:|------------------:|--------------------:|-------------------:|------------------:|---------------:|--------------------:|-----------------:|------------:|-----------------------:|---------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------|:-------------|:------------|-----------------:|
## |  0 |                  nan |                    nan |                              nan |                nan |                nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |                nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                        nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                          nan |                    nan |               nan | 3eba28e7-64c2-4e95-ac89-5d9948154c1d |       1 |                    nan |        nan |     303470 |        0 |                     nan |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |               nan |           nan |                     nan |             nan |            nan |                nan |              nan |                nan |           nan |              nan |                 nan |         nan |        1 | Regular Play   |      nan |        nan |            1 | Real Madrid       | nan                                      |        0 |              nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |                 nan |              nan |         nan |                    nan |                        nan | {'formation': 4141, 'lineup': [{'player': {'id': 3509, 'name': 'Thibaut Courtois'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 13}, {'player': {'id': 5721, 'name': 'Daniel Carvajal Ramos'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 2}, {'player': {'id': 5485, 'name': 'Raphaël Varane'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 5}, {'player': {'id': 5201, 'name': 'Sergio Ramos García'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 5552, 'name': 'Marcelo Vieira da Silva Júnior'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 12}, {'player': {'id': 5539, 'name': 'Carlos Henrique Casimiro'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 6773, 'name': 'Federico Santiago Valverde Dipetta'}, 'position': {'id': 12, 'name': 'Right Midfield'}, 'jersey_number': 15}, {'player': {'id': 4926, 'name': 'Francisco Román Alarcón Suárez'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 22}, {'player': {'id': 5574, 'name': 'Toni Kroos'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 8}, {'player': {'id': 18395, 'name': 'Vinícius José Paixão de Oliveira Júnior'}, 'position': {'id': 16, 'name': 'Left Midfield'}, 'jersey_number': 25}, {'player': {'id': 19677, 'name': 'Karim Benzema'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'jersey_number': 9}]} | Real Madrid | 00:00:00.000 | Starting XI |              nan |
## |  1 |                  nan |                    nan |                              nan |                nan |                nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |                nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                        nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                          nan |                    nan |               nan | e9a4b6da-b58f-4c56-a0e3-65bd23a17473 |       2 |                    nan |        nan |     303470 |        0 |                     nan |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |               nan |           nan |                     nan |             nan |            nan |                nan |              nan |                nan |           nan |              nan |                 nan |         nan |        1 | Regular Play   |      nan |        nan |            1 | Real Madrid       | nan                                      |        0 |              nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |                 nan |              nan |         nan |                    nan |                        nan | {'formation': 442, 'lineup': [{'player': {'id': 20055, 'name': 'Marc-André ter Stegen'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 6374, 'name': 'Nélson Cabral Semedo'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 2}, {'player': {'id': 5213, 'name': 'Gerard Piqué Bernabéu'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 3}, {'player': {'id': 5492, 'name': 'Samuel Yves Umtiti'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 23}, {'player': {'id': 5211, 'name': 'Jordi Alba Ramos'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 18}, {'player': {'id': 11392, 'name': 'Arthur Henrique Ramos de Oliveira Melo'}, 'position': {'id': 9, 'name': 'Right Defensive Midfield'}, 'jersey_number': 8}, {'player': {'id': 5203, 'name': 'Sergio Busquets i Burgos'}, 'position': {'id': 11, 'name': 'Left Defensive Midfield'}, 'jersey_number': 5}, {'player': {'id': 8206, 'name': 'Arturo Erasmo Vidal Pardo'}, 'position': {'id': 12, 'name': 'Right Midfield'}, 'jersey_number': 22}, {'player': {'id': 8118, 'name': 'Frenkie de Jong'}, 'position': {'id': 16, 'name': 'Left Midfield'}, 'jersey_number': 21}, {'player': {'id': 5503, 'name': 'Lionel Andrés Messi Cuccittini'}, 'position': {'id': 22, 'name': 'Right Center Forward'}, 'jersey_number': 10}, {'player': {'id': 5487, 'name': 'Antoine Griezmann'}, 'position': {'id': 24, 'name': 'Left Center Forward'}, 'jersey_number': 17}]}  | Barcelona   | 00:00:00.000 | Starting XI |              nan |
## |  2 |                  nan |                    nan |                              nan |                nan |                nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |                nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                        nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                          nan |                    nan |               nan | 0b6f982e-e92c-4732-bc96-b28176e28b28 |       3 |                    nan |        nan |     303470 |        0 |                     nan |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |               nan |           nan |                     nan |             nan |            nan |                nan |              nan |                nan |           nan |              nan |                 nan |         nan |        1 | Regular Play   |      nan |        nan |            1 | Real Madrid       | ['2e806582-2185-47ab-a971-b2898b602ea7'] |        0 |              nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Barcelona   | 00:00:00.000 | Half Start  |              nan |
## |  3 |                  nan |                    nan |                              nan |                nan |                nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |                nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                        nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                          nan |                    nan |               nan | 2e806582-2185-47ab-a971-b2898b602ea7 |       4 |                    nan |        nan |     303470 |        0 |                     nan |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |               nan |           nan |                     nan |             nan |            nan |                nan |              nan |                nan |           nan |              nan |                 nan |         nan |        1 | Regular Play   |      nan |        nan |            1 | Real Madrid       | ['0b6f982e-e92c-4732-bc96-b28176e28b28'] |        0 |              nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Real Madrid | 00:00:00.000 | Half Start  |              nan |
## |  4 |                  nan |                    nan |                              nan |                nan |                nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |                nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                        nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                          nan |                    nan |               nan | 91ccd2c5-6471-4fa1-aab6-b8fbe18d57af |    2410 |                    nan |        nan |     303470 |       45 |                     nan |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |              nan |                 nan |                nan |           nan |               nan |           nan |                     nan |             nan |            nan |                nan |              nan |                nan |           nan |              nan |                 nan |         nan |        2 | From Free Kick |      nan |        nan |          121 | Barcelona         | ['95da779c-a2df-4b0c-a486-661a30ccce63'] |        0 |              nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Barcelona   | 00:00:00.000 | Half Start  |              nan |
```

We can get an overview of all the different kinds of events that took place during the match, by looking intro the column names of the `events` dataset:


```python
print(events.columns)
```

```
## Index(['bad_behaviour_card', 'ball_receipt_outcome',
##        'ball_recovery_recovery_failure', 'block_deflection',
##        'block_save_block', 'carry_end_location', 'clearance_aerial_won',
##        'clearance_body_part', 'clearance_head', 'clearance_left_foot',
##        'clearance_right_foot', 'counterpress', 'dribble_no_touch',
##        'dribble_outcome', 'dribble_overrun', 'duel_outcome', 'duel_type',
##        'duration', 'foul_committed_advantage', 'foul_committed_card',
##        'foul_committed_offensive', 'foul_committed_type', 'foul_won_advantage',
##        'foul_won_defensive', 'goalkeeper_body_part', 'goalkeeper_end_location',
##        'goalkeeper_outcome', 'goalkeeper_position',
##        'goalkeeper_success_in_play', 'goalkeeper_technique', 'goalkeeper_type',
##        'id', 'index', 'interception_outcome', 'location', 'match_id', 'minute',
##        'miscontrol_aerial_won', 'off_camera', 'out', 'pass_aerial_won',
##        'pass_angle', 'pass_assisted_shot_id', 'pass_body_part', 'pass_cross',
##        'pass_cut_back', 'pass_deflected', 'pass_end_location',
##        'pass_goal_assist', 'pass_height', 'pass_inswinging', 'pass_length',
##        'pass_miscommunication', 'pass_no_touch', 'pass_outcome',
##        'pass_outswinging', 'pass_recipient', 'pass_shot_assist', 'pass_switch',
##        'pass_technique', 'pass_through_ball', 'pass_type', 'period',
##        'play_pattern', 'player', 'position', 'possession', 'possession_team',
##        'related_events', 'second', 'shot_body_part', 'shot_deflected',
##        'shot_end_location', 'shot_first_time', 'shot_freeze_frame',
##        'shot_key_pass_id', 'shot_one_on_one', 'shot_outcome',
##        'shot_statsbomb_xg', 'shot_technique', 'shot_type',
##        'substitution_outcome', 'substitution_replacement', 'tactics', 'team',
##        'timestamp', 'type', 'under_pressure'],
##       dtype='object')
```

In the first part of this tutorial we will focus on the passes played by a player during the match. It seems relevant information can be drawn if we filter out the following columns from the whole `events` dataset: `'team'`, `'type'`, `'minute'`, `'location'`, `'pass_end_location'`, `'pass_outcome'`, and `'player'`. We can easily do this in Python:


```python
events_pass = events[['team', 'type', 'minute', 'location', 'pass_end_location', 'pass_outcome', 'player']]
```

Let us look at the first and the last 10 rows of the `events_pass` dataset. We can do that by using the `head()` and `tail` function provided by `pandas`.


```python
e1 = events_pass.head(10) # extracts the first 10 rows
print(e1.to_markdown())
```

```
## |    | team        | type        |   minute | location     | pass_end_location   |   pass_outcome | player                         |
## |---:|:------------|:------------|---------:|:-------------|:--------------------|---------------:|:-------------------------------|
## |  0 | Real Madrid | Starting XI |        0 | nan          | nan                 |            nan | nan                            |
## |  1 | Barcelona   | Starting XI |        0 | nan          | nan                 |            nan | nan                            |
## |  2 | Barcelona   | Half Start  |        0 | nan          | nan                 |            nan | nan                            |
## |  3 | Real Madrid | Half Start  |        0 | nan          | nan                 |            nan | nan                            |
## |  4 | Barcelona   | Half Start  |       45 | nan          | nan                 |            nan | nan                            |
## |  5 | Real Madrid | Half Start  |       45 | nan          | nan                 |            nan | nan                            |
## |  6 | Real Madrid | Pass        |        0 | [60.0, 40.0] | [53.2, 43.5]        |            nan | Karim Benzema                  |
## |  7 | Real Madrid | Pass        |        0 | [53.3, 43.5] | [47.1, 36.9]        |            nan | Francisco Román Alarcón Suárez |
## |  8 | Real Madrid | Pass        |        0 | [47.1, 36.9] | [47.1, 31.0]        |            nan | Carlos Henrique Casimiro       |
## |  9 | Real Madrid | Pass        |        0 | [47.1, 31.0] | [47.4, 37.1]        |            nan | Toni Kroos                     |
```


```python
e2 = events_pass.tail(10) # extracts the last 10 rows
print(e2.to_markdown())
```

```
## |      | team        | type           |   minute |   location |   pass_end_location |   pass_outcome | player                                 |
## |-----:|:------------|:---------------|---------:|-----------:|--------------------:|---------------:|:---------------------------------------|
## | 4217 | Barcelona   | Half End       |       45 |        nan |                 nan |            nan | nan                                    |
## | 4218 | Real Madrid | Half End       |       93 |        nan |                 nan |            nan | nan                                    |
## | 4219 | Barcelona   | Half End       |       93 |        nan |                 nan |            nan | nan                                    |
## | 4220 | Barcelona   | Substitution   |       69 |        nan |                 nan |            nan | Arturo Erasmo Vidal Pardo              |
## | 4221 | Real Madrid | Substitution   |       78 |        nan |                 nan |            nan | Francisco Román Alarcón Suárez         |
## | 4222 | Barcelona   | Substitution   |       79 |        nan |                 nan |            nan | Arthur Henrique Ramos de Oliveira Melo |
## | 4223 | Barcelona   | Substitution   |       80 |        nan |                 nan |            nan | Antoine Griezmann                      |
## | 4224 | Real Madrid | Substitution   |       85 |        nan |                 nan |            nan | Federico Santiago Valverde Dipetta     |
## | 4225 | Real Madrid | Substitution   |       90 |        nan |                 nan |            nan | Karim Benzema                          |
## | 4226 | Barcelona   | Tactical Shift |       70 |        nan |                 nan |            nan | nan                                    |
```

Now, looking at both `e1` and `e2`, we notice that the `player` column gives us the names of the players who were associated with different events during the match. Suppose, we are only interested to generate the pass map and its corresponding heat map for a particular player, for example, `'Toni Kroos'`. For that, we have to clean the `events_pass` dataset in such a way that, we have only those rows where `player='Toni Kroos'`. Be very careful to use the exact spelling while performing these string operations, otherwise the reader will end up with unnecessary syntax and/or logical errors. Before filtering, let us collect the name of all the players who were involved in this match. For this, we use the `unique()` function provided by `pandas` that helps us extract a unique data from a column.


```python
players = events_pass.player.unique()
print(players)
```

```
## [nan 'Karim Benzema' 'Francisco Román Alarcón Suárez'
##  'Carlos Henrique Casimiro' 'Toni Kroos' 'Sergio Ramos García'
##  'Marcelo Vieira da Silva Júnior' 'Raphaël Varane' 'Daniel Carvajal Ramos'
##  'Vinícius José Paixão de Oliveira Júnior'
##  'Lionel Andrés Messi Cuccittini' 'Frenkie de Jong'
##  'Arthur Henrique Ramos de Oliveira Melo' 'Gerard Piqué Bernabéu'
##  'Arturo Erasmo Vidal Pardo' 'Sergio Busquets i Burgos'
##  'Nélson Cabral Semedo' 'Antoine Griezmann' 'Samuel Yves Umtiti'
##  'Jordi Alba Ramos' 'Federico Santiago Valverde Dipetta'
##  'Marc-André ter Stegen' 'Thibaut Courtois'
##  'Martin Braithwaite Christensen' 'Luka Modrić' 'Ivan Rakitić'
##  'Lucas Vázquez Iglesias' 'Anssumane Fati' 'Mariano Díaz Mejía']
```

We will now filter the dataset by the player name (`'Toni Kroos'` in our case). One good practice is to simply copy the particular player name from the `players` list that we just generated and use it according to our needs. This way, the spelling errors can be avoided. The filtration with python is an easy process:


```python
events_pass_p1 = events_pass[events_pass['player'] == 'Toni Kroos']
```

The first and the last 10 rows can be viewed again for the new `events_pass_p1` dataset:


```python
print(events_pass_p1.head(10).to_markdown())
```

```
## |     | team        | type   |   minute | location      | pass_end_location   | pass_outcome   | player     |
## |----:|:------------|:-------|---------:|:--------------|:--------------------|:---------------|:-----------|
## |   9 | Real Madrid | Pass   |        0 | [47.1, 31.0]  | [47.4, 37.1]        | nan            | Toni Kroos |
## |  11 | Real Madrid | Pass   |        0 | [46.6, 29.8]  | [47.1, 36.9]        | nan            | Toni Kroos |
## |  19 | Real Madrid | Pass   |        0 | [54.1, 47.7]  | [44.0, 48.6]        | nan            | Toni Kroos |
## |  24 | Real Madrid | Pass   |        0 | [68.4, 7.1]   | [69.1, 15.2]        | nan            | Toni Kroos |
## |  27 | Real Madrid | Pass   |        0 | [60.7, 6.3]   | [70.7, 1.6]         | nan            | Toni Kroos |
## |  32 | Real Madrid | Pass   |        1 | [115.7, 11.3] | [116.3, 12.2]       | Incomplete     | Toni Kroos |
## |  33 | Real Madrid | Pass   |        1 | [120.0, 0.1]  | [111.2, 35.3]       | Incomplete     | Toni Kroos |
## |  85 | Real Madrid | Pass   |        3 | [49.3, 5.4]   | [33.1, 24.3]        | nan            | Toni Kroos |
## | 103 | Real Madrid | Pass   |        4 | [43.6, 7.6]   | [34.8, 10.2]        | nan            | Toni Kroos |
## | 116 | Real Madrid | Pass   |        5 | [77.3, 6.0]   | [73.6, 0.7]         | nan            | Toni Kroos |
```


```python
print(events_pass_p1.tail(10).to_markdown())
```

```
## |      | team        | type           |   minute | location      |   pass_end_location |   pass_outcome | player     |
## |-----:|:------------|:---------------|---------:|:--------------|--------------------:|---------------:|:-----------|
## | 4034 | Real Madrid | Foul Committed |       17 | [52.4, 1.2]   |                 nan |            nan | Toni Kroos |
## | 4048 | Real Madrid | Foul Committed |       52 | [99.9, 14.1]  |                 nan |            nan | Toni Kroos |
## | 4057 | Real Madrid | Foul Committed |       84 | [47.0, 45.0]  |                 nan |            nan | Toni Kroos |
## | 4065 | Real Madrid | Duel           |       10 | [58.0, 41.0]  |                 nan |            nan | Toni Kroos |
## | 4071 | Real Madrid | Duel           |       18 | [42.1, 1.5]   |                 nan |            nan | Toni Kroos |
## | 4111 | Real Madrid | Shot           |       14 | [104.0, 33.0] |                 nan |            nan | Toni Kroos |
## | 4113 | Real Madrid | Shot           |       28 | [95.2, 42.1]  |                 nan |            nan | Toni Kroos |
## | 4125 | Real Madrid | Shot           |       61 | [93.3, 40.9]  |                 nan |            nan | Toni Kroos |
## | 4165 | Real Madrid | Dispossessed   |       40 | [90.2, 38.4]  |                 nan |            nan | Toni Kroos |
## | 4185 | Real Madrid | Foul Won       |       21 | [29.1, 2.1]   |                 nan |            nan | Toni Kroos |
```

Now, looking into both `e1` and `e2` our intuition tells us that the `type` column in `events_pass_p1` has event types other than passes, which we do not want for now. Thus, we have to again clean the dataset such that we have only those rows where `type = Pass`. The other rows can be discarded for now. Before that, let us analyse what event types other than 'Pass' are available for `'Toni Kroos'`:


```python
print(events_pass_p1.type.unique())
```

```
## ['Pass' 'Ball Receipt*' 'Carry' 'Pressure' 'Block' 'Ball Recovery'
##  'Interception' 'Dribbled Past' 'Miscontrol' 'Foul Committed' 'Duel'
##  'Shot' 'Dispossessed' 'Foul Won']
```

Seems our German maestro has been involved in a lot of events throughout the game. But let us focus on his passes foo now. We will again filter the dataset and reset its index so that the indexing restarts from `0`:


```python
events_pass_p1 = events_pass_p1[events_pass_p1['type'] == 'Pass'].reset_index()
print(events_pass_p1.to_markdown())
```

```
## |    |   index | team        | type   |   minute | location      | pass_end_location   | pass_outcome   | player     |
## |---:|--------:|:------------|:-------|---------:|:--------------|:--------------------|:---------------|:-----------|
## |  0 |       9 | Real Madrid | Pass   |        0 | [47.1, 31.0]  | [47.4, 37.1]        | nan            | Toni Kroos |
## |  1 |      11 | Real Madrid | Pass   |        0 | [46.6, 29.8]  | [47.1, 36.9]        | nan            | Toni Kroos |
## |  2 |      19 | Real Madrid | Pass   |        0 | [54.1, 47.7]  | [44.0, 48.6]        | nan            | Toni Kroos |
## |  3 |      24 | Real Madrid | Pass   |        0 | [68.4, 7.1]   | [69.1, 15.2]        | nan            | Toni Kroos |
## |  4 |      27 | Real Madrid | Pass   |        0 | [60.7, 6.3]   | [70.7, 1.6]         | nan            | Toni Kroos |
## |  5 |      32 | Real Madrid | Pass   |        1 | [115.7, 11.3] | [116.3, 12.2]       | Incomplete     | Toni Kroos |
## |  6 |      33 | Real Madrid | Pass   |        1 | [120.0, 0.1]  | [111.2, 35.3]       | Incomplete     | Toni Kroos |
## |  7 |      85 | Real Madrid | Pass   |        3 | [49.3, 5.4]   | [33.1, 24.3]        | nan            | Toni Kroos |
## |  8 |     103 | Real Madrid | Pass   |        4 | [43.6, 7.6]   | [34.8, 10.2]        | nan            | Toni Kroos |
## |  9 |     116 | Real Madrid | Pass   |        5 | [77.3, 6.0]   | [73.6, 0.7]         | nan            | Toni Kroos |
## | 10 |     126 | Real Madrid | Pass   |        6 | [120.0, 0.1]  | [100.0, 11.6]       | nan            | Toni Kroos |
## | 11 |     174 | Real Madrid | Pass   |        8 | [20.2, 28.4]  | [30.1, 53.9]        | nan            | Toni Kroos |
## | 12 |     177 | Real Madrid | Pass   |        9 | [34.2, 21.9]  | [48.1, 1.8]         | nan            | Toni Kroos |
## | 13 |     181 | Real Madrid | Pass   |        9 | [50.7, 21.9]  | [63.3, 66.2]        | nan            | Toni Kroos |
## | 14 |     189 | Real Madrid | Pass   |        9 | [70.3, 24.8]  | [90.7, 3.5]         | nan            | Toni Kroos |
## | 15 |     198 | Real Madrid | Pass   |       10 | [57.7, 43.5]  | [57.9, 54.1]        | nan            | Toni Kroos |
## | 16 |     212 | Real Madrid | Pass   |       11 | [60.7, 23.9]  | [76.6, 1.6]         | nan            | Toni Kroos |
## | 17 |     251 | Real Madrid | Pass   |       13 | [83.8, 22.2]  | [99.1, 11.0]        | nan            | Toni Kroos |
## | 18 |     253 | Real Madrid | Pass   |       13 | [90.5, 23.1]  | [102.9, 36.9]       | nan            | Toni Kroos |
## | 19 |     339 | Real Madrid | Pass   |       19 | [14.4, 66.5]  | [8.4, 75.2]         | nan            | Toni Kroos |
## | 20 |     342 | Real Madrid | Pass   |       19 | [23.2, 75.8]  | [7.7, 69.0]         | nan            | Toni Kroos |
## | 21 |     358 | Real Madrid | Pass   |       21 | [18.5, 22.0]  | [9.3, 23.4]         | nan            | Toni Kroos |
## | 22 |     360 | Real Madrid | Pass   |       21 | [33.6, 2.3]   | [24.7, 40.2]        | nan            | Toni Kroos |
## | 23 |     371 | Real Madrid | Pass   |       22 | [92.5, 19.7]  | [101.3, 18.9]       | nan            | Toni Kroos |
## | 24 |     384 | Real Madrid | Pass   |       22 | [77.0, 20.6]  | [84.3, 23.7]        | nan            | Toni Kroos |
## | 25 |     440 | Real Madrid | Pass   |       25 | [43.5, 44.2]  | [35.0, 50.0]        | nan            | Toni Kroos |
## | 26 |     448 | Real Madrid | Pass   |       25 | [59.5, 12.3]  | [70.5, 12.0]        | nan            | Toni Kroos |
## | 27 |     450 | Real Madrid | Pass   |       25 | [63.5, 18.4]  | [72.7, 22.3]        | nan            | Toni Kroos |
## | 28 |     452 | Real Madrid | Pass   |       25 | [65.6, 16.5]  | [69.2, 29.9]        | nan            | Toni Kroos |
## | 29 |     506 | Real Madrid | Pass   |       29 | [47.4, 6.0]   | [54.3, 0.5]         | nan            | Toni Kroos |
## | 30 |     510 | Real Madrid | Pass   |       29 | [47.9, 15.4]  | [51.5, 23.5]        | nan            | Toni Kroos |
## | 31 |     518 | Real Madrid | Pass   |       30 | [46.8, 35.3]  | [63.9, 6.2]         | nan            | Toni Kroos |
## | 32 |     522 | Real Madrid | Pass   |       30 | [62.5, 26.5]  | [79.6, 77.8]        | nan            | Toni Kroos |
## | 33 |     530 | Real Madrid | Pass   |       30 | [64.9, 52.6]  | [62.7, 66.2]        | nan            | Toni Kroos |
## | 34 |     533 | Real Madrid | Pass   |       30 | [66.3, 42.0]  | [72.1, 18.5]        | nan            | Toni Kroos |
## | 35 |     559 | Real Madrid | Pass   |       32 | [73.3, 12.7]  | [91.4, 1.8]         | nan            | Toni Kroos |
## | 36 |     621 | Real Madrid | Pass   |       38 | [38.8, 31.3]  | [47.4, 24.1]        | nan            | Toni Kroos |
## | 37 |     623 | Real Madrid | Pass   |       38 | [42.0, 26.7]  | [48.7, 11.2]        | nan            | Toni Kroos |
## | 38 |     626 | Real Madrid | Pass   |       38 | [57.1, 17.0]  | [56.3, 31.1]        | nan            | Toni Kroos |
## | 39 |     628 | Real Madrid | Pass   |       38 | [63.3, 18.7]  | [69.1, 30.1]        | nan            | Toni Kroos |
## | 40 |     630 | Real Madrid | Pass   |       38 | [68.1, 19.8]  | [64.3, 26.3]        | nan            | Toni Kroos |
## | 41 |     638 | Real Madrid | Pass   |       38 | [89.3, 17.9]  | [86.0, 25.1]        | nan            | Toni Kroos |
## | 42 |     650 | Real Madrid | Pass   |       39 | [44.8, 8.5]   | [49.0, 2.0]         | nan            | Toni Kroos |
## | 43 |     706 | Real Madrid | Pass   |       43 | [81.3, 4.8]   | [110.1, 46.6]       | Incomplete     | Toni Kroos |
## | 44 |     799 | Real Madrid | Pass   |       51 | [27.3, 22.4]  | [7.8, 38.3]         | nan            | Toni Kroos |
## | 45 |     804 | Real Madrid | Pass   |       51 | [33.2, 30.2]  | [43.0, 21.6]        | nan            | Toni Kroos |
## | 46 |     822 | Real Madrid | Pass   |       53 | [43.7, 47.5]  | [54.7, 47.1]        | nan            | Toni Kroos |
## | 47 |     840 | Real Madrid | Pass   |       54 | [73.3, 50.2]  | [84.4, 43.7]        | nan            | Toni Kroos |
## | 48 |     849 | Real Madrid | Pass   |       54 | [92.4, 29.4]  | [98.9, 23.0]        | nan            | Toni Kroos |
## | 49 |     851 | Real Madrid | Pass   |       54 | [112.6, 19.5] | [113.2, 33.0]       | Incomplete     | Toni Kroos |
## | 50 |     852 | Real Madrid | Pass   |       55 | [120.0, 0.1]  | [107.5, 16.1]       | nan            | Toni Kroos |
## | 51 |     854 | Real Madrid | Pass   |       55 | [120.0, 80.0] | [109.9, 46.7]       | Incomplete     | Toni Kroos |
## | 52 |     855 | Real Madrid | Pass   |       56 | [120.0, 80.0] | [116.6, 52.0]       | Incomplete     | Toni Kroos |
## | 53 |     873 | Real Madrid | Pass   |       57 | [54.1, 28.8]  | [47.9, 15.3]        | nan            | Toni Kroos |
## | 54 |     879 | Real Madrid | Pass   |       58 | [86.4, 24.2]  | [85.2, 40.3]        | nan            | Toni Kroos |
## | 55 |     889 | Real Madrid | Pass   |       59 | [80.1, 18.4]  | [98.3, 9.5]         | nan            | Toni Kroos |
## | 56 |     900 | Real Madrid | Pass   |       60 | [96.7, 26.6]  | [96.4, 56.9]        | nan            | Toni Kroos |
## | 57 |     993 | Real Madrid | Pass   |       68 | [120.0, 0.1]  | [110.8, 43.9]       | Incomplete     | Toni Kroos |
## | 58 |    1013 | Real Madrid | Pass   |       70 | [90.6, 5.9]   | [112.4, 13.3]       | nan            | Toni Kroos |
## | 59 |    1023 | Real Madrid | Pass   |       72 | [75.2, 22.4]  | [84.2, 2.4]         | nan            | Toni Kroos |
## | 60 |    1027 | Real Madrid | Pass   |       72 | [72.2, 9.6]   | [80.0, 17.2]        | nan            | Toni Kroos |
## | 61 |    1051 | Real Madrid | Pass   |       73 | [110.1, 24.4] | [111.4, 39.7]       | Incomplete     | Toni Kroos |
## | 62 |    1062 | Real Madrid | Pass   |       74 | [83.6, 30.8]  | [83.2, 40.5]        | nan            | Toni Kroos |
## | 63 |    1068 | Real Madrid | Pass   |       75 | [120.0, 80.0] | [99.0, 63.3]        | nan            | Toni Kroos |
## | 64 |    1077 | Real Madrid | Pass   |       76 | [40.8, 20.3]  | [47.5, 6.0]         | nan            | Toni Kroos |
## | 65 |    1086 | Real Madrid | Pass   |       76 | [86.0, 24.0]  | [83.2, 57.7]        | nan            | Toni Kroos |
## | 66 |    1089 | Real Madrid | Pass   |       76 | [75.4, 34.3]  | [72.2, 69.1]        | nan            | Toni Kroos |
## | 67 |    1100 | Real Madrid | Pass   |       77 | [65.4, 46.9]  | [73.4, 68.5]        | nan            | Toni Kroos |
## | 68 |    1117 | Real Madrid | Pass   |       79 | [79.6, 30.1]  | [77.0, 44.1]        | nan            | Toni Kroos |
## | 69 |    1123 | Real Madrid | Pass   |       79 | [70.4, 25.8]  | [64.2, 29.0]        | nan            | Toni Kroos |
## | 70 |    1171 | Real Madrid | Pass   |       84 | [22.1, 19.6]  | [13.3, 29.0]        | nan            | Toni Kroos |
## | 71 |    1178 | Real Madrid | Pass   |       85 | [55.6, 24.4]  | [116.3, 6.4]        | Pass Offside   | Toni Kroos |
## | 72 |    1187 | Real Madrid | Pass   |       86 | [64.8, 27.4]  | [41.7, 38.0]        | nan            | Toni Kroos |
```

So, till now, we have been successful in extracting out the pass event data for `'Toni Kroos'` from the match. That is a brilliant achievement to be honest. You deserve a pat on your back!

Getting back, we observe that `'Toni Kroos'` has been involved in `73` passes. We will later work out his pass success rate. But look at the number. Isn't he a brilliant midfielder that the German national team and the Real Madrid team have in their disposal? What a playmaker he is! Let us find out what were all his pass outcomes:


```python
print(events_pass_p1.pass_outcome.unique())
```

```
## [nan 'Incomplete' 'Pass Offside']
```

One important thing to notice is that, according to statsbomb data, `nan` as a pass outcome actually means a successful pass. Any other pass outcomes refer to unsuccessful passes. For `'Toni Kross'`, we notice that besides `nan` he has `Incomplete` and `Pass Offside` as pass outcomes. If we look closely the `events_pass_p1` dataframe has the `minute` column which tells us at what minute the pass had started from `Kroos`'s end. It also has the `location` and the `pass_end_location` columns informing us about the coordinates of `Kroos` when he pass the ball and the coordinates of where the ball ended after the pass (successful or unsuccessful). A successful pass means the ball ended with a player and the `pass_end_location` coordinates for this case gives the location of the player who received the ball. Both these columns have elements as lists of two numbers indicating the coordinates. Let us manipulate the `pass_outcome` column by replacing all the `nan` values with `'successful'` with the help of `fillna()` function provided by `pandas`. This will teach us the simplest way to handle `nan` values.


```python
events_pass_p1['pass_outcome'] = events_pass_p1['pass_outcome'].fillna('Successful')
print(events_pass_p1.to_markdown())
```

```
## |    |   index | team        | type   |   minute | location      | pass_end_location   | pass_outcome   | player     |
## |---:|--------:|:------------|:-------|---------:|:--------------|:--------------------|:---------------|:-----------|
## |  0 |       9 | Real Madrid | Pass   |        0 | [47.1, 31.0]  | [47.4, 37.1]        | Successful     | Toni Kroos |
## |  1 |      11 | Real Madrid | Pass   |        0 | [46.6, 29.8]  | [47.1, 36.9]        | Successful     | Toni Kroos |
## |  2 |      19 | Real Madrid | Pass   |        0 | [54.1, 47.7]  | [44.0, 48.6]        | Successful     | Toni Kroos |
## |  3 |      24 | Real Madrid | Pass   |        0 | [68.4, 7.1]   | [69.1, 15.2]        | Successful     | Toni Kroos |
## |  4 |      27 | Real Madrid | Pass   |        0 | [60.7, 6.3]   | [70.7, 1.6]         | Successful     | Toni Kroos |
## |  5 |      32 | Real Madrid | Pass   |        1 | [115.7, 11.3] | [116.3, 12.2]       | Incomplete     | Toni Kroos |
## |  6 |      33 | Real Madrid | Pass   |        1 | [120.0, 0.1]  | [111.2, 35.3]       | Incomplete     | Toni Kroos |
## |  7 |      85 | Real Madrid | Pass   |        3 | [49.3, 5.4]   | [33.1, 24.3]        | Successful     | Toni Kroos |
## |  8 |     103 | Real Madrid | Pass   |        4 | [43.6, 7.6]   | [34.8, 10.2]        | Successful     | Toni Kroos |
## |  9 |     116 | Real Madrid | Pass   |        5 | [77.3, 6.0]   | [73.6, 0.7]         | Successful     | Toni Kroos |
## | 10 |     126 | Real Madrid | Pass   |        6 | [120.0, 0.1]  | [100.0, 11.6]       | Successful     | Toni Kroos |
## | 11 |     174 | Real Madrid | Pass   |        8 | [20.2, 28.4]  | [30.1, 53.9]        | Successful     | Toni Kroos |
## | 12 |     177 | Real Madrid | Pass   |        9 | [34.2, 21.9]  | [48.1, 1.8]         | Successful     | Toni Kroos |
## | 13 |     181 | Real Madrid | Pass   |        9 | [50.7, 21.9]  | [63.3, 66.2]        | Successful     | Toni Kroos |
## | 14 |     189 | Real Madrid | Pass   |        9 | [70.3, 24.8]  | [90.7, 3.5]         | Successful     | Toni Kroos |
## | 15 |     198 | Real Madrid | Pass   |       10 | [57.7, 43.5]  | [57.9, 54.1]        | Successful     | Toni Kroos |
## | 16 |     212 | Real Madrid | Pass   |       11 | [60.7, 23.9]  | [76.6, 1.6]         | Successful     | Toni Kroos |
## | 17 |     251 | Real Madrid | Pass   |       13 | [83.8, 22.2]  | [99.1, 11.0]        | Successful     | Toni Kroos |
## | 18 |     253 | Real Madrid | Pass   |       13 | [90.5, 23.1]  | [102.9, 36.9]       | Successful     | Toni Kroos |
## | 19 |     339 | Real Madrid | Pass   |       19 | [14.4, 66.5]  | [8.4, 75.2]         | Successful     | Toni Kroos |
## | 20 |     342 | Real Madrid | Pass   |       19 | [23.2, 75.8]  | [7.7, 69.0]         | Successful     | Toni Kroos |
## | 21 |     358 | Real Madrid | Pass   |       21 | [18.5, 22.0]  | [9.3, 23.4]         | Successful     | Toni Kroos |
## | 22 |     360 | Real Madrid | Pass   |       21 | [33.6, 2.3]   | [24.7, 40.2]        | Successful     | Toni Kroos |
## | 23 |     371 | Real Madrid | Pass   |       22 | [92.5, 19.7]  | [101.3, 18.9]       | Successful     | Toni Kroos |
## | 24 |     384 | Real Madrid | Pass   |       22 | [77.0, 20.6]  | [84.3, 23.7]        | Successful     | Toni Kroos |
## | 25 |     440 | Real Madrid | Pass   |       25 | [43.5, 44.2]  | [35.0, 50.0]        | Successful     | Toni Kroos |
## | 26 |     448 | Real Madrid | Pass   |       25 | [59.5, 12.3]  | [70.5, 12.0]        | Successful     | Toni Kroos |
## | 27 |     450 | Real Madrid | Pass   |       25 | [63.5, 18.4]  | [72.7, 22.3]        | Successful     | Toni Kroos |
## | 28 |     452 | Real Madrid | Pass   |       25 | [65.6, 16.5]  | [69.2, 29.9]        | Successful     | Toni Kroos |
## | 29 |     506 | Real Madrid | Pass   |       29 | [47.4, 6.0]   | [54.3, 0.5]         | Successful     | Toni Kroos |
## | 30 |     510 | Real Madrid | Pass   |       29 | [47.9, 15.4]  | [51.5, 23.5]        | Successful     | Toni Kroos |
## | 31 |     518 | Real Madrid | Pass   |       30 | [46.8, 35.3]  | [63.9, 6.2]         | Successful     | Toni Kroos |
## | 32 |     522 | Real Madrid | Pass   |       30 | [62.5, 26.5]  | [79.6, 77.8]        | Successful     | Toni Kroos |
## | 33 |     530 | Real Madrid | Pass   |       30 | [64.9, 52.6]  | [62.7, 66.2]        | Successful     | Toni Kroos |
## | 34 |     533 | Real Madrid | Pass   |       30 | [66.3, 42.0]  | [72.1, 18.5]        | Successful     | Toni Kroos |
## | 35 |     559 | Real Madrid | Pass   |       32 | [73.3, 12.7]  | [91.4, 1.8]         | Successful     | Toni Kroos |
## | 36 |     621 | Real Madrid | Pass   |       38 | [38.8, 31.3]  | [47.4, 24.1]        | Successful     | Toni Kroos |
## | 37 |     623 | Real Madrid | Pass   |       38 | [42.0, 26.7]  | [48.7, 11.2]        | Successful     | Toni Kroos |
## | 38 |     626 | Real Madrid | Pass   |       38 | [57.1, 17.0]  | [56.3, 31.1]        | Successful     | Toni Kroos |
## | 39 |     628 | Real Madrid | Pass   |       38 | [63.3, 18.7]  | [69.1, 30.1]        | Successful     | Toni Kroos |
## | 40 |     630 | Real Madrid | Pass   |       38 | [68.1, 19.8]  | [64.3, 26.3]        | Successful     | Toni Kroos |
## | 41 |     638 | Real Madrid | Pass   |       38 | [89.3, 17.9]  | [86.0, 25.1]        | Successful     | Toni Kroos |
## | 42 |     650 | Real Madrid | Pass   |       39 | [44.8, 8.5]   | [49.0, 2.0]         | Successful     | Toni Kroos |
## | 43 |     706 | Real Madrid | Pass   |       43 | [81.3, 4.8]   | [110.1, 46.6]       | Incomplete     | Toni Kroos |
## | 44 |     799 | Real Madrid | Pass   |       51 | [27.3, 22.4]  | [7.8, 38.3]         | Successful     | Toni Kroos |
## | 45 |     804 | Real Madrid | Pass   |       51 | [33.2, 30.2]  | [43.0, 21.6]        | Successful     | Toni Kroos |
## | 46 |     822 | Real Madrid | Pass   |       53 | [43.7, 47.5]  | [54.7, 47.1]        | Successful     | Toni Kroos |
## | 47 |     840 | Real Madrid | Pass   |       54 | [73.3, 50.2]  | [84.4, 43.7]        | Successful     | Toni Kroos |
## | 48 |     849 | Real Madrid | Pass   |       54 | [92.4, 29.4]  | [98.9, 23.0]        | Successful     | Toni Kroos |
## | 49 |     851 | Real Madrid | Pass   |       54 | [112.6, 19.5] | [113.2, 33.0]       | Incomplete     | Toni Kroos |
## | 50 |     852 | Real Madrid | Pass   |       55 | [120.0, 0.1]  | [107.5, 16.1]       | Successful     | Toni Kroos |
## | 51 |     854 | Real Madrid | Pass   |       55 | [120.0, 80.0] | [109.9, 46.7]       | Incomplete     | Toni Kroos |
## | 52 |     855 | Real Madrid | Pass   |       56 | [120.0, 80.0] | [116.6, 52.0]       | Incomplete     | Toni Kroos |
## | 53 |     873 | Real Madrid | Pass   |       57 | [54.1, 28.8]  | [47.9, 15.3]        | Successful     | Toni Kroos |
## | 54 |     879 | Real Madrid | Pass   |       58 | [86.4, 24.2]  | [85.2, 40.3]        | Successful     | Toni Kroos |
## | 55 |     889 | Real Madrid | Pass   |       59 | [80.1, 18.4]  | [98.3, 9.5]         | Successful     | Toni Kroos |
## | 56 |     900 | Real Madrid | Pass   |       60 | [96.7, 26.6]  | [96.4, 56.9]        | Successful     | Toni Kroos |
## | 57 |     993 | Real Madrid | Pass   |       68 | [120.0, 0.1]  | [110.8, 43.9]       | Incomplete     | Toni Kroos |
## | 58 |    1013 | Real Madrid | Pass   |       70 | [90.6, 5.9]   | [112.4, 13.3]       | Successful     | Toni Kroos |
## | 59 |    1023 | Real Madrid | Pass   |       72 | [75.2, 22.4]  | [84.2, 2.4]         | Successful     | Toni Kroos |
## | 60 |    1027 | Real Madrid | Pass   |       72 | [72.2, 9.6]   | [80.0, 17.2]        | Successful     | Toni Kroos |
## | 61 |    1051 | Real Madrid | Pass   |       73 | [110.1, 24.4] | [111.4, 39.7]       | Incomplete     | Toni Kroos |
## | 62 |    1062 | Real Madrid | Pass   |       74 | [83.6, 30.8]  | [83.2, 40.5]        | Successful     | Toni Kroos |
## | 63 |    1068 | Real Madrid | Pass   |       75 | [120.0, 80.0] | [99.0, 63.3]        | Successful     | Toni Kroos |
## | 64 |    1077 | Real Madrid | Pass   |       76 | [40.8, 20.3]  | [47.5, 6.0]         | Successful     | Toni Kroos |
## | 65 |    1086 | Real Madrid | Pass   |       76 | [86.0, 24.0]  | [83.2, 57.7]        | Successful     | Toni Kroos |
## | 66 |    1089 | Real Madrid | Pass   |       76 | [75.4, 34.3]  | [72.2, 69.1]        | Successful     | Toni Kroos |
## | 67 |    1100 | Real Madrid | Pass   |       77 | [65.4, 46.9]  | [73.4, 68.5]        | Successful     | Toni Kroos |
## | 68 |    1117 | Real Madrid | Pass   |       79 | [79.6, 30.1]  | [77.0, 44.1]        | Successful     | Toni Kroos |
## | 69 |    1123 | Real Madrid | Pass   |       79 | [70.4, 25.8]  | [64.2, 29.0]        | Successful     | Toni Kroos |
## | 70 |    1171 | Real Madrid | Pass   |       84 | [22.1, 19.6]  | [13.3, 29.0]        | Successful     | Toni Kroos |
## | 71 |    1178 | Real Madrid | Pass   |       85 | [55.6, 24.4]  | [116.3, 6.4]        | Pass Offside   | Toni Kroos |
## | 72 |    1187 | Real Madrid | Pass   |       86 | [64.8, 27.4]  | [41.7, 38.0]        | Successful     | Toni Kroos |
```

Before diving into plotting the pass map and its heat map, let us complete the last bit of data manipulation. As the elements of `location` and the `pass_end_location` columns are lists, we will break each of the columns into two columns with the x-coordinates in one and y-coordinates in the other. Let us first carry out this manipulation for the `location` column:


```python
Loc = events_pass_p1['location']
Loc = pd.DataFrame(Loc.to_list(), columns=['location_x', 'location_y'])
print(Loc.to_markdown())
```

```
## |    |   location_x |   location_y |
## |---:|-------------:|-------------:|
## |  0 |         47.1 |         31   |
## |  1 |         46.6 |         29.8 |
## |  2 |         54.1 |         47.7 |
## |  3 |         68.4 |          7.1 |
## |  4 |         60.7 |          6.3 |
## |  5 |        115.7 |         11.3 |
## |  6 |        120   |          0.1 |
## |  7 |         49.3 |          5.4 |
## |  8 |         43.6 |          7.6 |
## |  9 |         77.3 |          6   |
## | 10 |        120   |          0.1 |
## | 11 |         20.2 |         28.4 |
## | 12 |         34.2 |         21.9 |
## | 13 |         50.7 |         21.9 |
## | 14 |         70.3 |         24.8 |
## | 15 |         57.7 |         43.5 |
## | 16 |         60.7 |         23.9 |
## | 17 |         83.8 |         22.2 |
## | 18 |         90.5 |         23.1 |
## | 19 |         14.4 |         66.5 |
## | 20 |         23.2 |         75.8 |
## | 21 |         18.5 |         22   |
## | 22 |         33.6 |          2.3 |
## | 23 |         92.5 |         19.7 |
## | 24 |         77   |         20.6 |
## | 25 |         43.5 |         44.2 |
## | 26 |         59.5 |         12.3 |
## | 27 |         63.5 |         18.4 |
## | 28 |         65.6 |         16.5 |
## | 29 |         47.4 |          6   |
## | 30 |         47.9 |         15.4 |
## | 31 |         46.8 |         35.3 |
## | 32 |         62.5 |         26.5 |
## | 33 |         64.9 |         52.6 |
## | 34 |         66.3 |         42   |
## | 35 |         73.3 |         12.7 |
## | 36 |         38.8 |         31.3 |
## | 37 |         42   |         26.7 |
## | 38 |         57.1 |         17   |
## | 39 |         63.3 |         18.7 |
## | 40 |         68.1 |         19.8 |
## | 41 |         89.3 |         17.9 |
## | 42 |         44.8 |          8.5 |
## | 43 |         81.3 |          4.8 |
## | 44 |         27.3 |         22.4 |
## | 45 |         33.2 |         30.2 |
## | 46 |         43.7 |         47.5 |
## | 47 |         73.3 |         50.2 |
## | 48 |         92.4 |         29.4 |
## | 49 |        112.6 |         19.5 |
## | 50 |        120   |          0.1 |
## | 51 |        120   |         80   |
## | 52 |        120   |         80   |
## | 53 |         54.1 |         28.8 |
## | 54 |         86.4 |         24.2 |
## | 55 |         80.1 |         18.4 |
## | 56 |         96.7 |         26.6 |
## | 57 |        120   |          0.1 |
## | 58 |         90.6 |          5.9 |
## | 59 |         75.2 |         22.4 |
## | 60 |         72.2 |          9.6 |
## | 61 |        110.1 |         24.4 |
## | 62 |         83.6 |         30.8 |
## | 63 |        120   |         80   |
## | 64 |         40.8 |         20.3 |
## | 65 |         86   |         24   |
## | 66 |         75.4 |         34.3 |
## | 67 |         65.4 |         46.9 |
## | 68 |         79.6 |         30.1 |
## | 69 |         70.4 |         25.8 |
## | 70 |         22.1 |         19.6 |
## | 71 |         55.6 |         24.4 |
## | 72 |         64.8 |         27.4 |
```

We see we have divided the `location` column into `location_x` and `location_y` columns. Let us operate the above chunk of code on the `pass_end_location` column too:


```python
Loc_end = events_pass_p1['pass_end_location']
Loc_end = pd.DataFrame(Loc_end.to_list(), columns=['pass_end_location_x', 'pass_end_location_y'])
print(Loc_end.to_markdown())
```

```
## |    |   pass_end_location_x |   pass_end_location_y |
## |---:|----------------------:|----------------------:|
## |  0 |                  47.4 |                  37.1 |
## |  1 |                  47.1 |                  36.9 |
## |  2 |                  44   |                  48.6 |
## |  3 |                  69.1 |                  15.2 |
## |  4 |                  70.7 |                   1.6 |
## |  5 |                 116.3 |                  12.2 |
## |  6 |                 111.2 |                  35.3 |
## |  7 |                  33.1 |                  24.3 |
## |  8 |                  34.8 |                  10.2 |
## |  9 |                  73.6 |                   0.7 |
## | 10 |                 100   |                  11.6 |
## | 11 |                  30.1 |                  53.9 |
## | 12 |                  48.1 |                   1.8 |
## | 13 |                  63.3 |                  66.2 |
## | 14 |                  90.7 |                   3.5 |
## | 15 |                  57.9 |                  54.1 |
## | 16 |                  76.6 |                   1.6 |
## | 17 |                  99.1 |                  11   |
## | 18 |                 102.9 |                  36.9 |
## | 19 |                   8.4 |                  75.2 |
## | 20 |                   7.7 |                  69   |
## | 21 |                   9.3 |                  23.4 |
## | 22 |                  24.7 |                  40.2 |
## | 23 |                 101.3 |                  18.9 |
## | 24 |                  84.3 |                  23.7 |
## | 25 |                  35   |                  50   |
## | 26 |                  70.5 |                  12   |
## | 27 |                  72.7 |                  22.3 |
## | 28 |                  69.2 |                  29.9 |
## | 29 |                  54.3 |                   0.5 |
## | 30 |                  51.5 |                  23.5 |
## | 31 |                  63.9 |                   6.2 |
## | 32 |                  79.6 |                  77.8 |
## | 33 |                  62.7 |                  66.2 |
## | 34 |                  72.1 |                  18.5 |
## | 35 |                  91.4 |                   1.8 |
## | 36 |                  47.4 |                  24.1 |
## | 37 |                  48.7 |                  11.2 |
## | 38 |                  56.3 |                  31.1 |
## | 39 |                  69.1 |                  30.1 |
## | 40 |                  64.3 |                  26.3 |
## | 41 |                  86   |                  25.1 |
## | 42 |                  49   |                   2   |
## | 43 |                 110.1 |                  46.6 |
## | 44 |                   7.8 |                  38.3 |
## | 45 |                  43   |                  21.6 |
## | 46 |                  54.7 |                  47.1 |
## | 47 |                  84.4 |                  43.7 |
## | 48 |                  98.9 |                  23   |
## | 49 |                 113.2 |                  33   |
## | 50 |                 107.5 |                  16.1 |
## | 51 |                 109.9 |                  46.7 |
## | 52 |                 116.6 |                  52   |
## | 53 |                  47.9 |                  15.3 |
## | 54 |                  85.2 |                  40.3 |
## | 55 |                  98.3 |                   9.5 |
## | 56 |                  96.4 |                  56.9 |
## | 57 |                 110.8 |                  43.9 |
## | 58 |                 112.4 |                  13.3 |
## | 59 |                  84.2 |                   2.4 |
## | 60 |                  80   |                  17.2 |
## | 61 |                 111.4 |                  39.7 |
## | 62 |                  83.2 |                  40.5 |
## | 63 |                  99   |                  63.3 |
## | 64 |                  47.5 |                   6   |
## | 65 |                  83.2 |                  57.7 |
## | 66 |                  72.2 |                  69.1 |
## | 67 |                  73.4 |                  68.5 |
## | 68 |                  77   |                  44.1 |
## | 69 |                  64.2 |                  29   |
## | 70 |                  13.3 |                  29   |
## | 71 |                 116.3 |                   6.4 |
## | 72 |                  41.7 |                  38   |
```

We will now add these columns to our `events_pass_1` data frame as extra columns and validate whether they are generating the correct coordinates:


```python
events_pass_p1['location_x'] = Loc['location_x']
events_pass_p1['location_y'] = Loc['location_y']
events_pass_p1['pass_end_location_x'] = Loc_end['pass_end_location_x']
events_pass_p1['pass_end_location_y'] = Loc_end['pass_end_location_y']
print(events_pass_p1.to_markdown())
```

```
## |    |   index | team        | type   |   minute | location      | pass_end_location   | pass_outcome   | player     |   location_x |   location_y |   pass_end_location_x |   pass_end_location_y |
## |---:|--------:|:------------|:-------|---------:|:--------------|:--------------------|:---------------|:-----------|-------------:|-------------:|----------------------:|----------------------:|
## |  0 |       9 | Real Madrid | Pass   |        0 | [47.1, 31.0]  | [47.4, 37.1]        | Successful     | Toni Kroos |         47.1 |         31   |                  47.4 |                  37.1 |
## |  1 |      11 | Real Madrid | Pass   |        0 | [46.6, 29.8]  | [47.1, 36.9]        | Successful     | Toni Kroos |         46.6 |         29.8 |                  47.1 |                  36.9 |
## |  2 |      19 | Real Madrid | Pass   |        0 | [54.1, 47.7]  | [44.0, 48.6]        | Successful     | Toni Kroos |         54.1 |         47.7 |                  44   |                  48.6 |
## |  3 |      24 | Real Madrid | Pass   |        0 | [68.4, 7.1]   | [69.1, 15.2]        | Successful     | Toni Kroos |         68.4 |          7.1 |                  69.1 |                  15.2 |
## |  4 |      27 | Real Madrid | Pass   |        0 | [60.7, 6.3]   | [70.7, 1.6]         | Successful     | Toni Kroos |         60.7 |          6.3 |                  70.7 |                   1.6 |
## |  5 |      32 | Real Madrid | Pass   |        1 | [115.7, 11.3] | [116.3, 12.2]       | Incomplete     | Toni Kroos |        115.7 |         11.3 |                 116.3 |                  12.2 |
## |  6 |      33 | Real Madrid | Pass   |        1 | [120.0, 0.1]  | [111.2, 35.3]       | Incomplete     | Toni Kroos |        120   |          0.1 |                 111.2 |                  35.3 |
## |  7 |      85 | Real Madrid | Pass   |        3 | [49.3, 5.4]   | [33.1, 24.3]        | Successful     | Toni Kroos |         49.3 |          5.4 |                  33.1 |                  24.3 |
## |  8 |     103 | Real Madrid | Pass   |        4 | [43.6, 7.6]   | [34.8, 10.2]        | Successful     | Toni Kroos |         43.6 |          7.6 |                  34.8 |                  10.2 |
## |  9 |     116 | Real Madrid | Pass   |        5 | [77.3, 6.0]   | [73.6, 0.7]         | Successful     | Toni Kroos |         77.3 |          6   |                  73.6 |                   0.7 |
## | 10 |     126 | Real Madrid | Pass   |        6 | [120.0, 0.1]  | [100.0, 11.6]       | Successful     | Toni Kroos |        120   |          0.1 |                 100   |                  11.6 |
## | 11 |     174 | Real Madrid | Pass   |        8 | [20.2, 28.4]  | [30.1, 53.9]        | Successful     | Toni Kroos |         20.2 |         28.4 |                  30.1 |                  53.9 |
## | 12 |     177 | Real Madrid | Pass   |        9 | [34.2, 21.9]  | [48.1, 1.8]         | Successful     | Toni Kroos |         34.2 |         21.9 |                  48.1 |                   1.8 |
## | 13 |     181 | Real Madrid | Pass   |        9 | [50.7, 21.9]  | [63.3, 66.2]        | Successful     | Toni Kroos |         50.7 |         21.9 |                  63.3 |                  66.2 |
## | 14 |     189 | Real Madrid | Pass   |        9 | [70.3, 24.8]  | [90.7, 3.5]         | Successful     | Toni Kroos |         70.3 |         24.8 |                  90.7 |                   3.5 |
## | 15 |     198 | Real Madrid | Pass   |       10 | [57.7, 43.5]  | [57.9, 54.1]        | Successful     | Toni Kroos |         57.7 |         43.5 |                  57.9 |                  54.1 |
## | 16 |     212 | Real Madrid | Pass   |       11 | [60.7, 23.9]  | [76.6, 1.6]         | Successful     | Toni Kroos |         60.7 |         23.9 |                  76.6 |                   1.6 |
## | 17 |     251 | Real Madrid | Pass   |       13 | [83.8, 22.2]  | [99.1, 11.0]        | Successful     | Toni Kroos |         83.8 |         22.2 |                  99.1 |                  11   |
## | 18 |     253 | Real Madrid | Pass   |       13 | [90.5, 23.1]  | [102.9, 36.9]       | Successful     | Toni Kroos |         90.5 |         23.1 |                 102.9 |                  36.9 |
## | 19 |     339 | Real Madrid | Pass   |       19 | [14.4, 66.5]  | [8.4, 75.2]         | Successful     | Toni Kroos |         14.4 |         66.5 |                   8.4 |                  75.2 |
## | 20 |     342 | Real Madrid | Pass   |       19 | [23.2, 75.8]  | [7.7, 69.0]         | Successful     | Toni Kroos |         23.2 |         75.8 |                   7.7 |                  69   |
## | 21 |     358 | Real Madrid | Pass   |       21 | [18.5, 22.0]  | [9.3, 23.4]         | Successful     | Toni Kroos |         18.5 |         22   |                   9.3 |                  23.4 |
## | 22 |     360 | Real Madrid | Pass   |       21 | [33.6, 2.3]   | [24.7, 40.2]        | Successful     | Toni Kroos |         33.6 |          2.3 |                  24.7 |                  40.2 |
## | 23 |     371 | Real Madrid | Pass   |       22 | [92.5, 19.7]  | [101.3, 18.9]       | Successful     | Toni Kroos |         92.5 |         19.7 |                 101.3 |                  18.9 |
## | 24 |     384 | Real Madrid | Pass   |       22 | [77.0, 20.6]  | [84.3, 23.7]        | Successful     | Toni Kroos |         77   |         20.6 |                  84.3 |                  23.7 |
## | 25 |     440 | Real Madrid | Pass   |       25 | [43.5, 44.2]  | [35.0, 50.0]        | Successful     | Toni Kroos |         43.5 |         44.2 |                  35   |                  50   |
## | 26 |     448 | Real Madrid | Pass   |       25 | [59.5, 12.3]  | [70.5, 12.0]        | Successful     | Toni Kroos |         59.5 |         12.3 |                  70.5 |                  12   |
## | 27 |     450 | Real Madrid | Pass   |       25 | [63.5, 18.4]  | [72.7, 22.3]        | Successful     | Toni Kroos |         63.5 |         18.4 |                  72.7 |                  22.3 |
## | 28 |     452 | Real Madrid | Pass   |       25 | [65.6, 16.5]  | [69.2, 29.9]        | Successful     | Toni Kroos |         65.6 |         16.5 |                  69.2 |                  29.9 |
## | 29 |     506 | Real Madrid | Pass   |       29 | [47.4, 6.0]   | [54.3, 0.5]         | Successful     | Toni Kroos |         47.4 |          6   |                  54.3 |                   0.5 |
## | 30 |     510 | Real Madrid | Pass   |       29 | [47.9, 15.4]  | [51.5, 23.5]        | Successful     | Toni Kroos |         47.9 |         15.4 |                  51.5 |                  23.5 |
## | 31 |     518 | Real Madrid | Pass   |       30 | [46.8, 35.3]  | [63.9, 6.2]         | Successful     | Toni Kroos |         46.8 |         35.3 |                  63.9 |                   6.2 |
## | 32 |     522 | Real Madrid | Pass   |       30 | [62.5, 26.5]  | [79.6, 77.8]        | Successful     | Toni Kroos |         62.5 |         26.5 |                  79.6 |                  77.8 |
## | 33 |     530 | Real Madrid | Pass   |       30 | [64.9, 52.6]  | [62.7, 66.2]        | Successful     | Toni Kroos |         64.9 |         52.6 |                  62.7 |                  66.2 |
## | 34 |     533 | Real Madrid | Pass   |       30 | [66.3, 42.0]  | [72.1, 18.5]        | Successful     | Toni Kroos |         66.3 |         42   |                  72.1 |                  18.5 |
## | 35 |     559 | Real Madrid | Pass   |       32 | [73.3, 12.7]  | [91.4, 1.8]         | Successful     | Toni Kroos |         73.3 |         12.7 |                  91.4 |                   1.8 |
## | 36 |     621 | Real Madrid | Pass   |       38 | [38.8, 31.3]  | [47.4, 24.1]        | Successful     | Toni Kroos |         38.8 |         31.3 |                  47.4 |                  24.1 |
## | 37 |     623 | Real Madrid | Pass   |       38 | [42.0, 26.7]  | [48.7, 11.2]        | Successful     | Toni Kroos |         42   |         26.7 |                  48.7 |                  11.2 |
## | 38 |     626 | Real Madrid | Pass   |       38 | [57.1, 17.0]  | [56.3, 31.1]        | Successful     | Toni Kroos |         57.1 |         17   |                  56.3 |                  31.1 |
## | 39 |     628 | Real Madrid | Pass   |       38 | [63.3, 18.7]  | [69.1, 30.1]        | Successful     | Toni Kroos |         63.3 |         18.7 |                  69.1 |                  30.1 |
## | 40 |     630 | Real Madrid | Pass   |       38 | [68.1, 19.8]  | [64.3, 26.3]        | Successful     | Toni Kroos |         68.1 |         19.8 |                  64.3 |                  26.3 |
## | 41 |     638 | Real Madrid | Pass   |       38 | [89.3, 17.9]  | [86.0, 25.1]        | Successful     | Toni Kroos |         89.3 |         17.9 |                  86   |                  25.1 |
## | 42 |     650 | Real Madrid | Pass   |       39 | [44.8, 8.5]   | [49.0, 2.0]         | Successful     | Toni Kroos |         44.8 |          8.5 |                  49   |                   2   |
## | 43 |     706 | Real Madrid | Pass   |       43 | [81.3, 4.8]   | [110.1, 46.6]       | Incomplete     | Toni Kroos |         81.3 |          4.8 |                 110.1 |                  46.6 |
## | 44 |     799 | Real Madrid | Pass   |       51 | [27.3, 22.4]  | [7.8, 38.3]         | Successful     | Toni Kroos |         27.3 |         22.4 |                   7.8 |                  38.3 |
## | 45 |     804 | Real Madrid | Pass   |       51 | [33.2, 30.2]  | [43.0, 21.6]        | Successful     | Toni Kroos |         33.2 |         30.2 |                  43   |                  21.6 |
## | 46 |     822 | Real Madrid | Pass   |       53 | [43.7, 47.5]  | [54.7, 47.1]        | Successful     | Toni Kroos |         43.7 |         47.5 |                  54.7 |                  47.1 |
## | 47 |     840 | Real Madrid | Pass   |       54 | [73.3, 50.2]  | [84.4, 43.7]        | Successful     | Toni Kroos |         73.3 |         50.2 |                  84.4 |                  43.7 |
## | 48 |     849 | Real Madrid | Pass   |       54 | [92.4, 29.4]  | [98.9, 23.0]        | Successful     | Toni Kroos |         92.4 |         29.4 |                  98.9 |                  23   |
## | 49 |     851 | Real Madrid | Pass   |       54 | [112.6, 19.5] | [113.2, 33.0]       | Incomplete     | Toni Kroos |        112.6 |         19.5 |                 113.2 |                  33   |
## | 50 |     852 | Real Madrid | Pass   |       55 | [120.0, 0.1]  | [107.5, 16.1]       | Successful     | Toni Kroos |        120   |          0.1 |                 107.5 |                  16.1 |
## | 51 |     854 | Real Madrid | Pass   |       55 | [120.0, 80.0] | [109.9, 46.7]       | Incomplete     | Toni Kroos |        120   |         80   |                 109.9 |                  46.7 |
## | 52 |     855 | Real Madrid | Pass   |       56 | [120.0, 80.0] | [116.6, 52.0]       | Incomplete     | Toni Kroos |        120   |         80   |                 116.6 |                  52   |
## | 53 |     873 | Real Madrid | Pass   |       57 | [54.1, 28.8]  | [47.9, 15.3]        | Successful     | Toni Kroos |         54.1 |         28.8 |                  47.9 |                  15.3 |
## | 54 |     879 | Real Madrid | Pass   |       58 | [86.4, 24.2]  | [85.2, 40.3]        | Successful     | Toni Kroos |         86.4 |         24.2 |                  85.2 |                  40.3 |
## | 55 |     889 | Real Madrid | Pass   |       59 | [80.1, 18.4]  | [98.3, 9.5]         | Successful     | Toni Kroos |         80.1 |         18.4 |                  98.3 |                   9.5 |
## | 56 |     900 | Real Madrid | Pass   |       60 | [96.7, 26.6]  | [96.4, 56.9]        | Successful     | Toni Kroos |         96.7 |         26.6 |                  96.4 |                  56.9 |
## | 57 |     993 | Real Madrid | Pass   |       68 | [120.0, 0.1]  | [110.8, 43.9]       | Incomplete     | Toni Kroos |        120   |          0.1 |                 110.8 |                  43.9 |
## | 58 |    1013 | Real Madrid | Pass   |       70 | [90.6, 5.9]   | [112.4, 13.3]       | Successful     | Toni Kroos |         90.6 |          5.9 |                 112.4 |                  13.3 |
## | 59 |    1023 | Real Madrid | Pass   |       72 | [75.2, 22.4]  | [84.2, 2.4]         | Successful     | Toni Kroos |         75.2 |         22.4 |                  84.2 |                   2.4 |
## | 60 |    1027 | Real Madrid | Pass   |       72 | [72.2, 9.6]   | [80.0, 17.2]        | Successful     | Toni Kroos |         72.2 |          9.6 |                  80   |                  17.2 |
## | 61 |    1051 | Real Madrid | Pass   |       73 | [110.1, 24.4] | [111.4, 39.7]       | Incomplete     | Toni Kroos |        110.1 |         24.4 |                 111.4 |                  39.7 |
## | 62 |    1062 | Real Madrid | Pass   |       74 | [83.6, 30.8]  | [83.2, 40.5]        | Successful     | Toni Kroos |         83.6 |         30.8 |                  83.2 |                  40.5 |
## | 63 |    1068 | Real Madrid | Pass   |       75 | [120.0, 80.0] | [99.0, 63.3]        | Successful     | Toni Kroos |        120   |         80   |                  99   |                  63.3 |
## | 64 |    1077 | Real Madrid | Pass   |       76 | [40.8, 20.3]  | [47.5, 6.0]         | Successful     | Toni Kroos |         40.8 |         20.3 |                  47.5 |                   6   |
## | 65 |    1086 | Real Madrid | Pass   |       76 | [86.0, 24.0]  | [83.2, 57.7]        | Successful     | Toni Kroos |         86   |         24   |                  83.2 |                  57.7 |
## | 66 |    1089 | Real Madrid | Pass   |       76 | [75.4, 34.3]  | [72.2, 69.1]        | Successful     | Toni Kroos |         75.4 |         34.3 |                  72.2 |                  69.1 |
## | 67 |    1100 | Real Madrid | Pass   |       77 | [65.4, 46.9]  | [73.4, 68.5]        | Successful     | Toni Kroos |         65.4 |         46.9 |                  73.4 |                  68.5 |
## | 68 |    1117 | Real Madrid | Pass   |       79 | [79.6, 30.1]  | [77.0, 44.1]        | Successful     | Toni Kroos |         79.6 |         30.1 |                  77   |                  44.1 |
## | 69 |    1123 | Real Madrid | Pass   |       79 | [70.4, 25.8]  | [64.2, 29.0]        | Successful     | Toni Kroos |         70.4 |         25.8 |                  64.2 |                  29   |
## | 70 |    1171 | Real Madrid | Pass   |       84 | [22.1, 19.6]  | [13.3, 29.0]        | Successful     | Toni Kroos |         22.1 |         19.6 |                  13.3 |                  29   |
## | 71 |    1178 | Real Madrid | Pass   |       85 | [55.6, 24.4]  | [116.3, 6.4]        | Pass Offside   | Toni Kroos |         55.6 |         24.4 |                 116.3 |                   6.4 |
## | 72 |    1187 | Real Madrid | Pass   |       86 | [64.8, 27.4]  | [41.7, 38.0]        | Successful     | Toni Kroos |         64.8 |         27.4 |                  41.7 |                  38   |
```

Seems like they are generating the correct coordinate values and we ere able to split the original `location` and `pass_end_location` columns. So, we don't actually need the original columns and they can be discarded along with the columns `team`, `type` and `player`, because those would be `'Real Madrid'`, `'Pass'` and `'Toni Kroos'` respectively for all the rows in `events_pass_p1`. Although it doesnot matter in our analysis, but it is always a good practice to clean your data as much as possible!


```python
events_pass_p1 = events_pass_p1[['minute', 'location_x', 'location_y', 'pass_end_location_x', 'pass_end_location_y', 'pass_outcome']]
print(events_pass_p1.to_markdown())
```

```
## |    |   minute |   location_x |   location_y |   pass_end_location_x |   pass_end_location_y | pass_outcome   |
## |---:|---------:|-------------:|-------------:|----------------------:|----------------------:|:---------------|
## |  0 |        0 |         47.1 |         31   |                  47.4 |                  37.1 | Successful     |
## |  1 |        0 |         46.6 |         29.8 |                  47.1 |                  36.9 | Successful     |
## |  2 |        0 |         54.1 |         47.7 |                  44   |                  48.6 | Successful     |
## |  3 |        0 |         68.4 |          7.1 |                  69.1 |                  15.2 | Successful     |
## |  4 |        0 |         60.7 |          6.3 |                  70.7 |                   1.6 | Successful     |
## |  5 |        1 |        115.7 |         11.3 |                 116.3 |                  12.2 | Incomplete     |
## |  6 |        1 |        120   |          0.1 |                 111.2 |                  35.3 | Incomplete     |
## |  7 |        3 |         49.3 |          5.4 |                  33.1 |                  24.3 | Successful     |
## |  8 |        4 |         43.6 |          7.6 |                  34.8 |                  10.2 | Successful     |
## |  9 |        5 |         77.3 |          6   |                  73.6 |                   0.7 | Successful     |
## | 10 |        6 |        120   |          0.1 |                 100   |                  11.6 | Successful     |
## | 11 |        8 |         20.2 |         28.4 |                  30.1 |                  53.9 | Successful     |
## | 12 |        9 |         34.2 |         21.9 |                  48.1 |                   1.8 | Successful     |
## | 13 |        9 |         50.7 |         21.9 |                  63.3 |                  66.2 | Successful     |
## | 14 |        9 |         70.3 |         24.8 |                  90.7 |                   3.5 | Successful     |
## | 15 |       10 |         57.7 |         43.5 |                  57.9 |                  54.1 | Successful     |
## | 16 |       11 |         60.7 |         23.9 |                  76.6 |                   1.6 | Successful     |
## | 17 |       13 |         83.8 |         22.2 |                  99.1 |                  11   | Successful     |
## | 18 |       13 |         90.5 |         23.1 |                 102.9 |                  36.9 | Successful     |
## | 19 |       19 |         14.4 |         66.5 |                   8.4 |                  75.2 | Successful     |
## | 20 |       19 |         23.2 |         75.8 |                   7.7 |                  69   | Successful     |
## | 21 |       21 |         18.5 |         22   |                   9.3 |                  23.4 | Successful     |
## | 22 |       21 |         33.6 |          2.3 |                  24.7 |                  40.2 | Successful     |
## | 23 |       22 |         92.5 |         19.7 |                 101.3 |                  18.9 | Successful     |
## | 24 |       22 |         77   |         20.6 |                  84.3 |                  23.7 | Successful     |
## | 25 |       25 |         43.5 |         44.2 |                  35   |                  50   | Successful     |
## | 26 |       25 |         59.5 |         12.3 |                  70.5 |                  12   | Successful     |
## | 27 |       25 |         63.5 |         18.4 |                  72.7 |                  22.3 | Successful     |
## | 28 |       25 |         65.6 |         16.5 |                  69.2 |                  29.9 | Successful     |
## | 29 |       29 |         47.4 |          6   |                  54.3 |                   0.5 | Successful     |
## | 30 |       29 |         47.9 |         15.4 |                  51.5 |                  23.5 | Successful     |
## | 31 |       30 |         46.8 |         35.3 |                  63.9 |                   6.2 | Successful     |
## | 32 |       30 |         62.5 |         26.5 |                  79.6 |                  77.8 | Successful     |
## | 33 |       30 |         64.9 |         52.6 |                  62.7 |                  66.2 | Successful     |
## | 34 |       30 |         66.3 |         42   |                  72.1 |                  18.5 | Successful     |
## | 35 |       32 |         73.3 |         12.7 |                  91.4 |                   1.8 | Successful     |
## | 36 |       38 |         38.8 |         31.3 |                  47.4 |                  24.1 | Successful     |
## | 37 |       38 |         42   |         26.7 |                  48.7 |                  11.2 | Successful     |
## | 38 |       38 |         57.1 |         17   |                  56.3 |                  31.1 | Successful     |
## | 39 |       38 |         63.3 |         18.7 |                  69.1 |                  30.1 | Successful     |
## | 40 |       38 |         68.1 |         19.8 |                  64.3 |                  26.3 | Successful     |
## | 41 |       38 |         89.3 |         17.9 |                  86   |                  25.1 | Successful     |
## | 42 |       39 |         44.8 |          8.5 |                  49   |                   2   | Successful     |
## | 43 |       43 |         81.3 |          4.8 |                 110.1 |                  46.6 | Incomplete     |
## | 44 |       51 |         27.3 |         22.4 |                   7.8 |                  38.3 | Successful     |
## | 45 |       51 |         33.2 |         30.2 |                  43   |                  21.6 | Successful     |
## | 46 |       53 |         43.7 |         47.5 |                  54.7 |                  47.1 | Successful     |
## | 47 |       54 |         73.3 |         50.2 |                  84.4 |                  43.7 | Successful     |
## | 48 |       54 |         92.4 |         29.4 |                  98.9 |                  23   | Successful     |
## | 49 |       54 |        112.6 |         19.5 |                 113.2 |                  33   | Incomplete     |
## | 50 |       55 |        120   |          0.1 |                 107.5 |                  16.1 | Successful     |
## | 51 |       55 |        120   |         80   |                 109.9 |                  46.7 | Incomplete     |
## | 52 |       56 |        120   |         80   |                 116.6 |                  52   | Incomplete     |
## | 53 |       57 |         54.1 |         28.8 |                  47.9 |                  15.3 | Successful     |
## | 54 |       58 |         86.4 |         24.2 |                  85.2 |                  40.3 | Successful     |
## | 55 |       59 |         80.1 |         18.4 |                  98.3 |                   9.5 | Successful     |
## | 56 |       60 |         96.7 |         26.6 |                  96.4 |                  56.9 | Successful     |
## | 57 |       68 |        120   |          0.1 |                 110.8 |                  43.9 | Incomplete     |
## | 58 |       70 |         90.6 |          5.9 |                 112.4 |                  13.3 | Successful     |
## | 59 |       72 |         75.2 |         22.4 |                  84.2 |                   2.4 | Successful     |
## | 60 |       72 |         72.2 |          9.6 |                  80   |                  17.2 | Successful     |
## | 61 |       73 |        110.1 |         24.4 |                 111.4 |                  39.7 | Incomplete     |
## | 62 |       74 |         83.6 |         30.8 |                  83.2 |                  40.5 | Successful     |
## | 63 |       75 |        120   |         80   |                  99   |                  63.3 | Successful     |
## | 64 |       76 |         40.8 |         20.3 |                  47.5 |                   6   | Successful     |
## | 65 |       76 |         86   |         24   |                  83.2 |                  57.7 | Successful     |
## | 66 |       76 |         75.4 |         34.3 |                  72.2 |                  69.1 | Successful     |
## | 67 |       77 |         65.4 |         46.9 |                  73.4 |                  68.5 | Successful     |
## | 68 |       79 |         79.6 |         30.1 |                  77   |                  44.1 | Successful     |
## | 69 |       79 |         70.4 |         25.8 |                  64.2 |                  29   | Successful     |
## | 70 |       84 |         22.1 |         19.6 |                  13.3 |                  29   | Successful     |
## | 71 |       85 |         55.6 |         24.4 |                 116.3 |                   6.4 | Pass Offside   |
## | 72 |       86 |         64.8 |         27.4 |                  41.7 |                  38   | Successful     |
```

Thus, we have successfully cleaned and manipulated the data. Now we can go ahead and start plotting the pass map for `Toni Kroos` on a football pitch and also visualize its corresponding heat map. Before going through this part, the readers should make sure that they have studied my [second post](https://realsoccerexpand.netlify.app/post/visualize-a-pitch/) on how to draw football pitches using the `mplsoccer` package. Let us draw a simple pitch first:


```python
pitch = Pitch(pitch_color = 'black', line_color = 'white', constrained_layout = True, tight_layout = False, goal_type = 'box')
fig, ax = pitch.draw()
plt.show()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-1.png" width="672" />

We now have a pitch on which we will plot the start and end coordinates of the passes and join them with lines having arrows to denote the correct direction of the passes. We will use a for loop to go through each row and extract the coordinates: 


```python
pitch = Pitch(pitch_color = 'black', line_color = 'white', constrained_layout = True, tight_layout = False, goal_type = 'box')
fig, ax = pitch.draw()

for i in range(len(events_pass_p1)):
    if events_pass_p1.pass_outcome[i] == 'Successful':
        pitch.arrows(events_pass_p1.location_x[i], events_pass_p1.location_y[i], events_pass_p1.pass_end_location_x[i], events_pass_p1.pass_end_location_y[i], ax=ax, color='green', width = 3)
        pitch.scatter(events_pass_p1.location_x[i], events_pass_p1.location_y[i], ax = ax, color = 'green')
    else:
        pitch.arrows(events_pass_p1.location_x[i], events_pass_p1.location_y[i], events_pass_p1.pass_end_location_x[i], events_pass_p1.pass_end_location_y[i], ax=ax, color='red', width=3)
        pitch.scatter(events_pass_p1.location_x[i], events_pass_p1.location_y[i], ax = ax, color='red')
```

```python
plt.show()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-3.png" width="672" />

The `pitch.arrows()` function allows us to draw the lines along with their arrow heads and the `pitch.scatter()` function allows us the pin point the starting location of the pass by plotting filled dots. the `width` parameter sets the widths of the lines and the `color` parameter sets the colors. The successful passes are colored `green` and the unsuccessful passes are colored `red`.

This makes us visually analyze `Kroos`'s passing abilities. Look at the number of successful passes he created during the match. Now we will focus on visualizing the heat map on the pass map of `'Toni Kroos'`. We will use `seaborn`'s `kdeplot()` function which portrays the probability density function of the data given by a contour plot showing the relationship of the data distribution between two data variables, here the pass start locations. `kdeplot` stands for *Kernel Distribution Estimation Plot*. Note that we have imported the `seaborn` package as `sns`. Let us draw the heatmap:


```python
# Pitch drawing code
pitch = Pitch(pitch_color = 'black', line_color = 'white', constrained_layout = True, tight_layout = False, goal_type = 'box')
fig, ax = pitch.draw()

# Heat map code
res = sns.kdeplot(events_pass_p1['location_x'], events_pass_p1['location_y'], fill = True,
    thresh = 0.05, alpha = 0.5, levels = 10, cmap = 'Purples_d')

# Pass map code
```

```python
for i in range(len(events_pass_p1)):
    if events_pass_p1.pass_outcome[i] == 'Successful':
        pitch.arrows(events_pass_p1.location_x[i], events_pass_p1.location_y[i], events_pass_p1.pass_end_location_x[i], events_pass_p1.pass_end_location_y[i], ax=ax, color='green', width = 3)
        pitch.scatter(events_pass_p1.location_x[i], events_pass_p1.location_y[i], ax = ax, color = 'green')
    else:
        pitch.arrows(events_pass_p1.location_x[i], events_pass_p1.location_y[i], events_pass_p1.pass_end_location_x[i], events_pass_p1.pass_end_location_y[i], ax=ax, color='red', width=3)
        pitch.scatter(events_pass_p1.location_x[i], events_pass_p1.location_y[i], ax = ax, color='red')
        
# General plot code
```

```python
plt.title("Toni Kroos pass and heat map")
```

```python
plt.show()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-5.png" width="672" />

The player is supposed to score a goal on the right side. So `Kroos` moves from the left to the right (remember the axis ranges from the second post). It seems he played most of his passes on his left side on the pitch where the density plot looks condensed. The passes that he took from the corners had both successful and unsuccessful outcomes. Looking into the arguments of the `kdeplot()` function, the `thresh` value sets the lowest iso-proportion level at which the contour lines are to be drawn, `levels` sets the number of contour levels, `fill` sets whether to fill the area between the contours,  the `alpha` sets the transparency of the plot (default value is `1`, lesser than `1` means more transparent), and the `cmap` sets the color map. To study more about `kdeplot()` look [here](https://seaborn.pydata.org/generated/seaborn.kdeplot.html).

Finally, for `'Toni Kroos'` let us calculate the percentage of successful and unsuccessful passes.


```python
events_pass_p1['pass_outcome'].value_counts(normalize=True).mul(100)
```

```
## Successful      87.671233
## Incomplete      10.958904
## Pass Offside     1.369863
## Name: pass_outcome, dtype: float64
```

To plot the frequency distribution:


```python
events_pass_p1['pass_outcome'].value_counts(normalize=True).mul(100).plot.bar()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-7.png" width="672" />

We notice that `'Kroos'` had created around 87.67% of successful passes. Wild!

We have completed our study on generating the pass map and its corresponding heat map for a particular player. The reader is suggested to go ahead and try out the same analysis for other players from the same game, for example for `'Lionel Andrés Messi Cuccittini'` (our king Messi) or for `'Luka Modrić'` (our all time favorite Lukita). For those who does not know, both of them are Ballon d'Or winners. So, I can guarantee the readers will be ecstatic to analyze the pass maps for these players. The readers can also compare the pass maps between two players from opposition teams or visualize the pass maps for a particular team (although this might look clumsy. Hint: skip the step where the filtration by the player's name is happening). A lot of customizations are possible. In the next part we will perform the same analysis on the shots instead of passes and we will visualize the shot maps and their corresponding heat maps for both the teams.

Let us create a new dataset by extracting out the relevant information on shot events from the original `events` dataset.


```python
events_shot = events[['team', 'type', 'minute', 'location', 'shot_end_location', 'shot_outcome', 'player']]
```

We can check how the `events_shot` dataset looks by printing its first and last 10 rows:


```python
print(events_shot.head(10).to_markdown())
```

```
## |    | team        | type        |   minute | location     |   shot_end_location |   shot_outcome | player                         |
## |---:|:------------|:------------|---------:|:-------------|--------------------:|---------------:|:-------------------------------|
## |  0 | Real Madrid | Starting XI |        0 | nan          |                 nan |            nan | nan                            |
## |  1 | Barcelona   | Starting XI |        0 | nan          |                 nan |            nan | nan                            |
## |  2 | Barcelona   | Half Start  |        0 | nan          |                 nan |            nan | nan                            |
## |  3 | Real Madrid | Half Start  |        0 | nan          |                 nan |            nan | nan                            |
## |  4 | Barcelona   | Half Start  |       45 | nan          |                 nan |            nan | nan                            |
## |  5 | Real Madrid | Half Start  |       45 | nan          |                 nan |            nan | nan                            |
## |  6 | Real Madrid | Pass        |        0 | [60.0, 40.0] |                 nan |            nan | Karim Benzema                  |
## |  7 | Real Madrid | Pass        |        0 | [53.3, 43.5] |                 nan |            nan | Francisco Román Alarcón Suárez |
## |  8 | Real Madrid | Pass        |        0 | [47.1, 36.9] |                 nan |            nan | Carlos Henrique Casimiro       |
## |  9 | Real Madrid | Pass        |        0 | [47.1, 31.0] |                 nan |            nan | Toni Kroos                     |
```


```python
print(events_shot.tail(10).to_markdown())
```

```
## |      | team        | type           |   minute |   location |   shot_end_location |   shot_outcome | player                                 |
## |-----:|:------------|:---------------|---------:|-----------:|--------------------:|---------------:|:---------------------------------------|
## | 4217 | Barcelona   | Half End       |       45 |        nan |                 nan |            nan | nan                                    |
## | 4218 | Real Madrid | Half End       |       93 |        nan |                 nan |            nan | nan                                    |
## | 4219 | Barcelona   | Half End       |       93 |        nan |                 nan |            nan | nan                                    |
## | 4220 | Barcelona   | Substitution   |       69 |        nan |                 nan |            nan | Arturo Erasmo Vidal Pardo              |
## | 4221 | Real Madrid | Substitution   |       78 |        nan |                 nan |            nan | Francisco Román Alarcón Suárez         |
## | 4222 | Barcelona   | Substitution   |       79 |        nan |                 nan |            nan | Arthur Henrique Ramos de Oliveira Melo |
## | 4223 | Barcelona   | Substitution   |       80 |        nan |                 nan |            nan | Antoine Griezmann                      |
## | 4224 | Real Madrid | Substitution   |       85 |        nan |                 nan |            nan | Federico Santiago Valverde Dipetta     |
## | 4225 | Real Madrid | Substitution   |       90 |        nan |                 nan |            nan | Karim Benzema                          |
## | 4226 | Barcelona   | Tactical Shift |       70 |        nan |                 nan |            nan | nan                                    |
```

Let us first filter the dataset by setting the event type to be `Shot`, that is only consider those rows where the column `type` is set as `'Shot'`. During our last analysis, we set this as `'Pass'`


```python
events_shot = events_shot[events_shot['type']  == 'Shot']
print(events_shot.to_markdown())
```

```
## |      | team        | type   |   minute | location      | shot_end_location   | shot_outcome   | player                                  |
## |-----:|:------------|:-------|---------:|:--------------|:--------------------|:---------------|:----------------------------------------|
## | 4110 | Real Madrid | Shot   |        6 | [104.5, 51.0] | [120.0, 42.7, 4.2]  | Off T          | Karim Benzema                           |
## | 4111 | Real Madrid | Shot   |       14 | [104.0, 33.0] | [120.0, 35.6, 4.6]  | Off T          | Toni Kroos                              |
## | 4112 | Barcelona   | Shot   |       20 | [108.5, 38.5] | [120.0, 36.3, 4.0]  | Off T          | Antoine Griezmann                       |
## | 4113 | Real Madrid | Shot   |       28 | [95.2, 42.1]  | [120.0, 41.8, 4.8]  | Off T          | Toni Kroos                              |
## | 4114 | Barcelona   | Shot   |       29 | [107.6, 53.6] | [117.7, 43.3, 0.2]  | Saved          | Lionel Andrés Messi Cuccittini          |
## | 4115 | Real Madrid | Shot   |       33 | [112.6, 39.1] | [114.4, 49.3]       | Wayward        | Karim Benzema                           |
## | 4116 | Barcelona   | Shot   |       33 | [109.0, 27.1] | [111.2, 30.2, 0.7]  | Saved          | Arthur Henrique Ramos de Oliveira Melo  |
## | 4117 | Barcelona   | Shot   |       36 | [100.8, 37.4] | [101.9, 37.7]       | Blocked        | Antoine Griezmann                       |
## | 4118 | Real Madrid | Shot   |       36 | [103.9, 26.1] | [117.3, 38.6, 1.6]  | Saved          | Vinícius José Paixão de Oliveira Júnior |
## | 4119 | Barcelona   | Shot   |       37 | [102.7, 40.7] | [115.1, 40.0, 1.4]  | Saved          | Lionel Andrés Messi Cuccittini          |
## | 4120 | Barcelona   | Shot   |       39 | [102.6, 42.7] | [103.8, 42.5]       | Blocked        | Arturo Erasmo Vidal Pardo               |
## | 4121 | Barcelona   | Shot   |       46 | [102.8, 26.2] | [103.7, 26.9]       | Blocked        | Lionel Andrés Messi Cuccittini          |
## | 4122 | Real Madrid | Shot   |       55 | [100.6, 24.0] | [119.4, 43.0, 2.2]  | Saved          | Francisco Román Alarcón Suárez          |
## | 4123 | Real Madrid | Shot   |       57 | [94.5, 30.4]  | [120.0, 49.7, 6.7]  | Off T          | Carlos Henrique Casimiro                |
## | 4124 | Real Madrid | Shot   |       60 | [112.3, 38.4] | [119.5, 40.7]       | Blocked        | Francisco Román Alarcón Suárez          |
## | 4125 | Real Madrid | Shot   |       61 | [93.3, 40.9]  | [120.0, 45.3, 3.9]  | Off T          | Toni Kroos                              |
## | 4126 | Real Madrid | Shot   |       62 | [114.1, 49.7] | [120.0, 38.3, 3.8]  | Off T          | Karim Benzema                           |
## | 4127 | Barcelona   | Shot   |       69 | [111.1, 57.5] | [112.4, 54.5, 0.4]  | Saved          | Martin Braithwaite Christensen          |
## | 4128 | Real Madrid | Shot   |       70 | [115.8, 27.3] | [120.0, 37.8, 0.4]  | Goal           | Vinícius José Paixão de Oliveira Júnior |
## | 4129 | Barcelona   | Shot   |       82 | [111.9, 34.7] | [120.0, 34.8, 3.8]  | Off T          | Gerard Piqué Bernabéu                   |
## | 4130 | Real Madrid | Shot   |       88 | [99.7, 39.1]  | [109.6, 38.2]       | Blocked        | Sergio Ramos García                     |
## | 4131 | Real Madrid | Shot   |       91 | [117.6, 50.1] | [120.0, 39.4, 0.2]  | Goal           | Mariano Díaz Mejía                      |
```

If we now closely examine `events_shot`, we see that it has the following relevant columns:

* `team`: the team the player who took the shot belonged to,
* `type`: it is set to ''Shot'` at all rows because we have filtered the dataset by this type,
* `minute`: the minute when the shot took place,
* `location`: the location of the ball where the shot was taken on the pitch. It is a list of numbers denoting the coordinates,
* `shot_end_location`: the location of the ball where it ended after shot was taken. It is a list of numbers denoting the coordinates,
* `shot_outcome`: the outcome of a particular shot, and 
* `player`: the player who took the shot.

Let us look into the different outcomes of the shots during the match:


```python
print(events_shot.shot_outcome.unique())
```

```
## ['Off T' 'Saved' 'Wayward' 'Blocked' 'Goal']
```

Now, like we have done in case of passes, we will again split the `location` and `shot_end_location` into two columns each with their x-coordinates in one and the y-coordinates in the other.


```python
shot_Loc = events_shot['location']
shot_Loc = pd.DataFrame(shot_Loc.to_list(), columns=['location_x', 'location_y'])
print(shot_Loc.to_markdown())
```

```
## |    |   location_x |   location_y |
## |---:|-------------:|-------------:|
## |  0 |        104.5 |         51   |
## |  1 |        104   |         33   |
## |  2 |        108.5 |         38.5 |
## |  3 |         95.2 |         42.1 |
## |  4 |        107.6 |         53.6 |
## |  5 |        112.6 |         39.1 |
## |  6 |        109   |         27.1 |
## |  7 |        100.8 |         37.4 |
## |  8 |        103.9 |         26.1 |
## |  9 |        102.7 |         40.7 |
## | 10 |        102.6 |         42.7 |
## | 11 |        102.8 |         26.2 |
## | 12 |        100.6 |         24   |
## | 13 |         94.5 |         30.4 |
## | 14 |        112.3 |         38.4 |
## | 15 |         93.3 |         40.9 |
## | 16 |        114.1 |         49.7 |
## | 17 |        111.1 |         57.5 |
## | 18 |        115.8 |         27.3 |
## | 19 |        111.9 |         34.7 |
## | 20 |         99.7 |         39.1 |
## | 21 |        117.6 |         50.1 |
```


```python
shot_end_Loc = events_shot['shot_end_location']
shot_end_Loc = pd.DataFrame(shot_end_Loc.to_list(), columns=['shot_end_location_x', 'shot_end_location_y', 'shot_end_location_z'])
print(shot_end_Loc.to_markdown())
```

```
## |    |   shot_end_location_x |   shot_end_location_y |   shot_end_location_z |
## |---:|----------------------:|----------------------:|----------------------:|
## |  0 |                 120   |                  42.7 |                   4.2 |
## |  1 |                 120   |                  35.6 |                   4.6 |
## |  2 |                 120   |                  36.3 |                   4   |
## |  3 |                 120   |                  41.8 |                   4.8 |
## |  4 |                 117.7 |                  43.3 |                   0.2 |
## |  5 |                 114.4 |                  49.3 |                 nan   |
## |  6 |                 111.2 |                  30.2 |                   0.7 |
## |  7 |                 101.9 |                  37.7 |                 nan   |
## |  8 |                 117.3 |                  38.6 |                   1.6 |
## |  9 |                 115.1 |                  40   |                   1.4 |
## | 10 |                 103.8 |                  42.5 |                 nan   |
## | 11 |                 103.7 |                  26.9 |                 nan   |
## | 12 |                 119.4 |                  43   |                   2.2 |
## | 13 |                 120   |                  49.7 |                   6.7 |
## | 14 |                 119.5 |                  40.7 |                 nan   |
## | 15 |                 120   |                  45.3 |                   3.9 |
## | 16 |                 120   |                  38.3 |                   3.8 |
## | 17 |                 112.4 |                  54.5 |                   0.4 |
## | 18 |                 120   |                  37.8 |                   0.4 |
## | 19 |                 120   |                  34.8 |                   3.8 |
## | 20 |                 109.6 |                  38.2 |                 nan   |
## | 21 |                 120   |                  39.4 |                   0.2 |
```

So, we can add these rows to our `events_shot` dataframe.


```python
events_shot = events_shot.reset_index()
events_shot['location_x'] = shot_Loc['location_x']
events_shot['location_y'] = shot_Loc['location_y']
events_shot['shot_end_location_x'] = shot_end_Loc['shot_end_location_x']
events_shot['shot_end_location_y'] = shot_end_Loc['shot_end_location_y']
print(events_shot.to_markdown())
```

```
## |    |   index | team        | type   |   minute | location      | shot_end_location   | shot_outcome   | player                                  |   location_x |   location_y |   shot_end_location_x |   shot_end_location_y |
## |---:|--------:|:------------|:-------|---------:|:--------------|:--------------------|:---------------|:----------------------------------------|-------------:|-------------:|----------------------:|----------------------:|
## |  0 |    4110 | Real Madrid | Shot   |        6 | [104.5, 51.0] | [120.0, 42.7, 4.2]  | Off T          | Karim Benzema                           |        104.5 |         51   |                 120   |                  42.7 |
## |  1 |    4111 | Real Madrid | Shot   |       14 | [104.0, 33.0] | [120.0, 35.6, 4.6]  | Off T          | Toni Kroos                              |        104   |         33   |                 120   |                  35.6 |
## |  2 |    4112 | Barcelona   | Shot   |       20 | [108.5, 38.5] | [120.0, 36.3, 4.0]  | Off T          | Antoine Griezmann                       |        108.5 |         38.5 |                 120   |                  36.3 |
## |  3 |    4113 | Real Madrid | Shot   |       28 | [95.2, 42.1]  | [120.0, 41.8, 4.8]  | Off T          | Toni Kroos                              |         95.2 |         42.1 |                 120   |                  41.8 |
## |  4 |    4114 | Barcelona   | Shot   |       29 | [107.6, 53.6] | [117.7, 43.3, 0.2]  | Saved          | Lionel Andrés Messi Cuccittini          |        107.6 |         53.6 |                 117.7 |                  43.3 |
## |  5 |    4115 | Real Madrid | Shot   |       33 | [112.6, 39.1] | [114.4, 49.3]       | Wayward        | Karim Benzema                           |        112.6 |         39.1 |                 114.4 |                  49.3 |
## |  6 |    4116 | Barcelona   | Shot   |       33 | [109.0, 27.1] | [111.2, 30.2, 0.7]  | Saved          | Arthur Henrique Ramos de Oliveira Melo  |        109   |         27.1 |                 111.2 |                  30.2 |
## |  7 |    4117 | Barcelona   | Shot   |       36 | [100.8, 37.4] | [101.9, 37.7]       | Blocked        | Antoine Griezmann                       |        100.8 |         37.4 |                 101.9 |                  37.7 |
## |  8 |    4118 | Real Madrid | Shot   |       36 | [103.9, 26.1] | [117.3, 38.6, 1.6]  | Saved          | Vinícius José Paixão de Oliveira Júnior |        103.9 |         26.1 |                 117.3 |                  38.6 |
## |  9 |    4119 | Barcelona   | Shot   |       37 | [102.7, 40.7] | [115.1, 40.0, 1.4]  | Saved          | Lionel Andrés Messi Cuccittini          |        102.7 |         40.7 |                 115.1 |                  40   |
## | 10 |    4120 | Barcelona   | Shot   |       39 | [102.6, 42.7] | [103.8, 42.5]       | Blocked        | Arturo Erasmo Vidal Pardo               |        102.6 |         42.7 |                 103.8 |                  42.5 |
## | 11 |    4121 | Barcelona   | Shot   |       46 | [102.8, 26.2] | [103.7, 26.9]       | Blocked        | Lionel Andrés Messi Cuccittini          |        102.8 |         26.2 |                 103.7 |                  26.9 |
## | 12 |    4122 | Real Madrid | Shot   |       55 | [100.6, 24.0] | [119.4, 43.0, 2.2]  | Saved          | Francisco Román Alarcón Suárez          |        100.6 |         24   |                 119.4 |                  43   |
## | 13 |    4123 | Real Madrid | Shot   |       57 | [94.5, 30.4]  | [120.0, 49.7, 6.7]  | Off T          | Carlos Henrique Casimiro                |         94.5 |         30.4 |                 120   |                  49.7 |
## | 14 |    4124 | Real Madrid | Shot   |       60 | [112.3, 38.4] | [119.5, 40.7]       | Blocked        | Francisco Román Alarcón Suárez          |        112.3 |         38.4 |                 119.5 |                  40.7 |
## | 15 |    4125 | Real Madrid | Shot   |       61 | [93.3, 40.9]  | [120.0, 45.3, 3.9]  | Off T          | Toni Kroos                              |         93.3 |         40.9 |                 120   |                  45.3 |
## | 16 |    4126 | Real Madrid | Shot   |       62 | [114.1, 49.7] | [120.0, 38.3, 3.8]  | Off T          | Karim Benzema                           |        114.1 |         49.7 |                 120   |                  38.3 |
## | 17 |    4127 | Barcelona   | Shot   |       69 | [111.1, 57.5] | [112.4, 54.5, 0.4]  | Saved          | Martin Braithwaite Christensen          |        111.1 |         57.5 |                 112.4 |                  54.5 |
## | 18 |    4128 | Real Madrid | Shot   |       70 | [115.8, 27.3] | [120.0, 37.8, 0.4]  | Goal           | Vinícius José Paixão de Oliveira Júnior |        115.8 |         27.3 |                 120   |                  37.8 |
## | 19 |    4129 | Barcelona   | Shot   |       82 | [111.9, 34.7] | [120.0, 34.8, 3.8]  | Off T          | Gerard Piqué Bernabéu                   |        111.9 |         34.7 |                 120   |                  34.8 |
## | 20 |    4130 | Real Madrid | Shot   |       88 | [99.7, 39.1]  | [109.6, 38.2]       | Blocked        | Sergio Ramos García                     |         99.7 |         39.1 |                 109.6 |                  38.2 |
## | 21 |    4131 | Real Madrid | Shot   |       91 | [117.6, 50.1] | [120.0, 39.4, 0.2]  | Goal           | Mariano Díaz Mejía                      |        117.6 |         50.1 |                 120   |                  39.4 |
```

We can now discard the unnecessary columns:


```python
events_shot = events_shot[['team', 'minute', 'player', 'location_x', 'location_y', 'shot_end_location_x', 'shot_end_location_y', 'shot_outcome']]
print(events_shot.to_markdown())
```

```
## |    | team        |   minute | player                                  |   location_x |   location_y |   shot_end_location_x |   shot_end_location_y | shot_outcome   |
## |---:|:------------|---------:|:----------------------------------------|-------------:|-------------:|----------------------:|----------------------:|:---------------|
## |  0 | Real Madrid |        6 | Karim Benzema                           |        104.5 |         51   |                 120   |                  42.7 | Off T          |
## |  1 | Real Madrid |       14 | Toni Kroos                              |        104   |         33   |                 120   |                  35.6 | Off T          |
## |  2 | Barcelona   |       20 | Antoine Griezmann                       |        108.5 |         38.5 |                 120   |                  36.3 | Off T          |
## |  3 | Real Madrid |       28 | Toni Kroos                              |         95.2 |         42.1 |                 120   |                  41.8 | Off T          |
## |  4 | Barcelona   |       29 | Lionel Andrés Messi Cuccittini          |        107.6 |         53.6 |                 117.7 |                  43.3 | Saved          |
## |  5 | Real Madrid |       33 | Karim Benzema                           |        112.6 |         39.1 |                 114.4 |                  49.3 | Wayward        |
## |  6 | Barcelona   |       33 | Arthur Henrique Ramos de Oliveira Melo  |        109   |         27.1 |                 111.2 |                  30.2 | Saved          |
## |  7 | Barcelona   |       36 | Antoine Griezmann                       |        100.8 |         37.4 |                 101.9 |                  37.7 | Blocked        |
## |  8 | Real Madrid |       36 | Vinícius José Paixão de Oliveira Júnior |        103.9 |         26.1 |                 117.3 |                  38.6 | Saved          |
## |  9 | Barcelona   |       37 | Lionel Andrés Messi Cuccittini          |        102.7 |         40.7 |                 115.1 |                  40   | Saved          |
## | 10 | Barcelona   |       39 | Arturo Erasmo Vidal Pardo               |        102.6 |         42.7 |                 103.8 |                  42.5 | Blocked        |
## | 11 | Barcelona   |       46 | Lionel Andrés Messi Cuccittini          |        102.8 |         26.2 |                 103.7 |                  26.9 | Blocked        |
## | 12 | Real Madrid |       55 | Francisco Román Alarcón Suárez          |        100.6 |         24   |                 119.4 |                  43   | Saved          |
## | 13 | Real Madrid |       57 | Carlos Henrique Casimiro                |         94.5 |         30.4 |                 120   |                  49.7 | Off T          |
## | 14 | Real Madrid |       60 | Francisco Román Alarcón Suárez          |        112.3 |         38.4 |                 119.5 |                  40.7 | Blocked        |
## | 15 | Real Madrid |       61 | Toni Kroos                              |         93.3 |         40.9 |                 120   |                  45.3 | Off T          |
## | 16 | Real Madrid |       62 | Karim Benzema                           |        114.1 |         49.7 |                 120   |                  38.3 | Off T          |
## | 17 | Barcelona   |       69 | Martin Braithwaite Christensen          |        111.1 |         57.5 |                 112.4 |                  54.5 | Saved          |
## | 18 | Real Madrid |       70 | Vinícius José Paixão de Oliveira Júnior |        115.8 |         27.3 |                 120   |                  37.8 | Goal           |
## | 19 | Barcelona   |       82 | Gerard Piqué Bernabéu                   |        111.9 |         34.7 |                 120   |                  34.8 | Off T          |
## | 20 | Real Madrid |       88 | Sergio Ramos García                     |         99.7 |         39.1 |                 109.6 |                  38.2 | Blocked        |
## | 21 | Real Madrid |       91 | Mariano Díaz Mejía                      |        117.6 |         50.1 |                 120   |                  39.4 | Goal           |
```

We have been again successful in cleaning and manipulating our dataset optimally according to our needs. We will now split the datset into two datasets, one for `Real Madrid` and the other for `Barcelona`.


```python
events_shot_Real = events_shot[events_shot['team'] == 'Real Madrid']
print(events_shot_Real.to_markdown())
```

```
## |    | team        |   minute | player                                  |   location_x |   location_y |   shot_end_location_x |   shot_end_location_y | shot_outcome   |
## |---:|:------------|---------:|:----------------------------------------|-------------:|-------------:|----------------------:|----------------------:|:---------------|
## |  0 | Real Madrid |        6 | Karim Benzema                           |        104.5 |         51   |                 120   |                  42.7 | Off T          |
## |  1 | Real Madrid |       14 | Toni Kroos                              |        104   |         33   |                 120   |                  35.6 | Off T          |
## |  3 | Real Madrid |       28 | Toni Kroos                              |         95.2 |         42.1 |                 120   |                  41.8 | Off T          |
## |  5 | Real Madrid |       33 | Karim Benzema                           |        112.6 |         39.1 |                 114.4 |                  49.3 | Wayward        |
## |  8 | Real Madrid |       36 | Vinícius José Paixão de Oliveira Júnior |        103.9 |         26.1 |                 117.3 |                  38.6 | Saved          |
## | 12 | Real Madrid |       55 | Francisco Román Alarcón Suárez          |        100.6 |         24   |                 119.4 |                  43   | Saved          |
## | 13 | Real Madrid |       57 | Carlos Henrique Casimiro                |         94.5 |         30.4 |                 120   |                  49.7 | Off T          |
## | 14 | Real Madrid |       60 | Francisco Román Alarcón Suárez          |        112.3 |         38.4 |                 119.5 |                  40.7 | Blocked        |
## | 15 | Real Madrid |       61 | Toni Kroos                              |         93.3 |         40.9 |                 120   |                  45.3 | Off T          |
## | 16 | Real Madrid |       62 | Karim Benzema                           |        114.1 |         49.7 |                 120   |                  38.3 | Off T          |
## | 18 | Real Madrid |       70 | Vinícius José Paixão de Oliveira Júnior |        115.8 |         27.3 |                 120   |                  37.8 | Goal           |
## | 20 | Real Madrid |       88 | Sergio Ramos García                     |         99.7 |         39.1 |                 109.6 |                  38.2 | Blocked        |
## | 21 | Real Madrid |       91 | Mariano Díaz Mejía                      |        117.6 |         50.1 |                 120   |                  39.4 | Goal           |
```


```python
events_shot_Barca = events_shot[events_shot['team'] == 'Barcelona']
print(events_shot_Barca.to_markdown())
```

```
## |    | team      |   minute | player                                 |   location_x |   location_y |   shot_end_location_x |   shot_end_location_y | shot_outcome   |
## |---:|:----------|---------:|:---------------------------------------|-------------:|-------------:|----------------------:|----------------------:|:---------------|
## |  2 | Barcelona |       20 | Antoine Griezmann                      |        108.5 |         38.5 |                 120   |                  36.3 | Off T          |
## |  4 | Barcelona |       29 | Lionel Andrés Messi Cuccittini         |        107.6 |         53.6 |                 117.7 |                  43.3 | Saved          |
## |  6 | Barcelona |       33 | Arthur Henrique Ramos de Oliveira Melo |        109   |         27.1 |                 111.2 |                  30.2 | Saved          |
## |  7 | Barcelona |       36 | Antoine Griezmann                      |        100.8 |         37.4 |                 101.9 |                  37.7 | Blocked        |
## |  9 | Barcelona |       37 | Lionel Andrés Messi Cuccittini         |        102.7 |         40.7 |                 115.1 |                  40   | Saved          |
## | 10 | Barcelona |       39 | Arturo Erasmo Vidal Pardo              |        102.6 |         42.7 |                 103.8 |                  42.5 | Blocked        |
## | 11 | Barcelona |       46 | Lionel Andrés Messi Cuccittini         |        102.8 |         26.2 |                 103.7 |                  26.9 | Blocked        |
## | 17 | Barcelona |       69 | Martin Braithwaite Christensen         |        111.1 |         57.5 |                 112.4 |                  54.5 | Saved          |
## | 19 | Barcelona |       82 | Gerard Piqué Bernabéu                  |        111.9 |         34.7 |                 120   |                  34.8 | Off T          |
```

So we have all the ingredients for visualization of the shot maps along with their heat maps for both the teams. We have already seen the unique types of shot outcomes and we will color code them accordingly. Let us first visualize the shot map and its corresponding heat map for `Real Madrid`. Note that we could have plotted the maps in a similar fashion to the one we did for passes. But let us make the exercise a little sophiticated by not using a for loop. Only caveat is that we now have to re-filter the dataset `events_shot_Real` by the outcomes of the shots. Let us first check what the shot outcomes are for `Real Madrid`:


```python
print(events_shot_Real.shot_outcome.unique())
```

```
## ['Off T' 'Wayward' 'Saved' 'Blocked' 'Goal']
```

So let us filter `events_shot_Real` by `shot_outcome` and generate separate datasets:


```python
events_shot_Real_Goal = events_shot_Real[events_shot_Real['shot_outcome'] == 'Goal']
events_shot_Real_off_wayward = events_shot_Real[(events_shot_Real['shot_outcome'] == 'Off T') | (events_shot_Real['shot_outcome'] == 'Wayward')]
events_shot_Real_saved_blocked = events_shot_Real[(events_shot_Real['shot_outcome'] == 'Saved') | (events_shot_Real['shot_outcome'] == 'Blocked')]

print(events_shot_Real_Goal.to_markdown())
```

```
## |    | team        |   minute | player                                  |   location_x |   location_y |   shot_end_location_x |   shot_end_location_y | shot_outcome   |
## |---:|:------------|---------:|:----------------------------------------|-------------:|-------------:|----------------------:|----------------------:|:---------------|
## | 18 | Real Madrid |       70 | Vinícius José Paixão de Oliveira Júnior |        115.8 |         27.3 |                   120 |                  37.8 | Goal           |
## | 21 | Real Madrid |       91 | Mariano Díaz Mejía                      |        117.6 |         50.1 |                   120 |                  39.4 | Goal           |
```

```python
print(events_shot_Real_off_wayward.to_markdown())
```

```
## |    | team        |   minute | player                   |   location_x |   location_y |   shot_end_location_x |   shot_end_location_y | shot_outcome   |
## |---:|:------------|---------:|:-------------------------|-------------:|-------------:|----------------------:|----------------------:|:---------------|
## |  0 | Real Madrid |        6 | Karim Benzema            |        104.5 |         51   |                 120   |                  42.7 | Off T          |
## |  1 | Real Madrid |       14 | Toni Kroos               |        104   |         33   |                 120   |                  35.6 | Off T          |
## |  3 | Real Madrid |       28 | Toni Kroos               |         95.2 |         42.1 |                 120   |                  41.8 | Off T          |
## |  5 | Real Madrid |       33 | Karim Benzema            |        112.6 |         39.1 |                 114.4 |                  49.3 | Wayward        |
## | 13 | Real Madrid |       57 | Carlos Henrique Casimiro |         94.5 |         30.4 |                 120   |                  49.7 | Off T          |
## | 15 | Real Madrid |       61 | Toni Kroos               |         93.3 |         40.9 |                 120   |                  45.3 | Off T          |
## | 16 | Real Madrid |       62 | Karim Benzema            |        114.1 |         49.7 |                 120   |                  38.3 | Off T          |
```

Now let us plot the pitch and the shot and heat maps:


```python
# Pitch drawing code
pitch = Pitch(pitch_color = 'black', line_color = 'white', constrained_layout = True, tight_layout = False, goal_type = 'box')
fig, ax = pitch.draw()

# Heat map code
res = sns.kdeplot(events_shot_Real['location_x'], events_shot_Real['location_y'], fill = True, 
thresh = 0.05, alpha = 0.5, levels = 10, cmap = 'Purples_d')

# Pass map code
```

```python
pitch.arrows(events_shot_Real_Goal.location_x, events_shot_Real_Goal.location_y, events_shot_Real_Goal.shot_end_location_x, events_shot_Real_Goal.shot_end_location_y, ax=ax, color='green', width = 3, label = 'Goals')
```

```python
pitch.scatter(events_shot_Real_Goal.location_x, events_shot_Real_Goal.location_y, ax = ax, color = 'green')
```

```python
pitch.arrows(events_shot_Real_off_wayward.location_x, events_shot_Real_off_wayward.location_y, events_shot_Real_off_wayward.shot_end_location_x, events_shot_Real_off_wayward.shot_end_location_y, ax=ax, color='red', width = 3, label = 'Off T / Wayward')
```

```python
pitch.scatter(events_shot_Real_off_wayward.location_x, events_shot_Real_off_wayward.location_y, ax = ax, color = 'red')
```

```python
pitch.arrows(events_shot_Real_saved_blocked.location_x, events_shot_Real_saved_blocked.location_y, events_shot_Real_saved_blocked.shot_end_location_x, events_shot_Real_saved_blocked.shot_end_location_y, ax=ax, color='orange', width = 3, label = 'Saved / Blocked')
```

```python
pitch.scatter(events_shot_Real_saved_blocked.location_x, events_shot_Real_saved_blocked.location_y, ax = ax, color = 'orange')

# General plot code
```

```python
ax.legend(handlelength = 3, edgecolor='None', fontsize = 10)
```

```python
plt.title("Real Madrid shot and heat map")
```

```python
plt.show()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-9.png" width="672" />

As we already know `Real Madrid` won the game `2-0` against `Barcelona`. This fact is evident by the two `green` arrows depicting the goals scored by `Real Madrid`. We will perform the same operations on the `events_shot_Barca` dataset. Let us first check what the shot outcomes are for `Barcelona`:


```python
print(events_shot_Barca.shot_outcome.unique())
```

```
## ['Off T' 'Saved' 'Blocked']
```

In case of `Barcelona` we will take the data manipulations a little further. Keep in mind that this is an optional but recommended step. Let us remind ourselves that in case of `statsbomb` pitch type, that we are currently using by default, the x-axis ranges from `0` to `120` (refer to the [second post](https://realsoccerexpand.netlify.app/post/visualize-a-pitch/)). Note that the statsbomb data lists the data coordinates in such a way that they are placed only on one half of the pitch, no matter which team we are focusing on. Although we are visualizing our maps separately for both the teams, it is recommended to subtract the values of `location_x` and `shot_location_x` from 120 for any one of the teams (NOT both!!!!). This will just invert the x-axis for one of the teams which will give us a better idea of the locations. Let us perform this operation on the `shots_event_Barsa` dataset.


```python
events_shot_Barca['location_x'] = 120 - events_shot_Barca['location_x']
```

```
## C:/Users/indra/AppData/Local/r-miniconda/envs/r-reticulate/python.exe:1: SettingWithCopyWarning: 
## A value is trying to be set on a copy of a slice from a DataFrame.
## Try using .loc[row_indexer,col_indexer] = value instead
## 
## See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
```

```python
events_shot_Barca['shot_end_location_x'] = 120 - events_shot_Barca['shot_end_location_x']
```

Let us again operate the same filtration by `shot_outcome` on the `events_shot_Barca` dataset:


```python
events_shot_Barca_off = events_shot_Barca[events_shot_Barca['shot_outcome'] == 'Off T']
events_shot_Barca_saved_blocked = events_shot_Barca[(events_shot_Barca['shot_outcome'] == 'Saved') | (events_shot_Barca['shot_outcome'] == 'Blocked')]

print(events_shot_Barca_off.to_markdown())
```

```
## |    | team      |   minute | player                |   location_x |   location_y |   shot_end_location_x |   shot_end_location_y | shot_outcome   |
## |---:|:----------|---------:|:----------------------|-------------:|-------------:|----------------------:|----------------------:|:---------------|
## |  2 | Barcelona |       20 | Antoine Griezmann     |         11.5 |         38.5 |                     0 |                  36.3 | Off T          |
## | 19 | Barcelona |       82 | Gerard Piqué Bernabéu |          8.1 |         34.7 |                     0 |                  34.8 | Off T          |
```

```python
print(events_shot_Barca_saved_blocked.to_markdown())
```

```
## |    | team      |   minute | player                                 |   location_x |   location_y |   shot_end_location_x |   shot_end_location_y | shot_outcome   |
## |---:|:----------|---------:|:---------------------------------------|-------------:|-------------:|----------------------:|----------------------:|:---------------|
## |  4 | Barcelona |       29 | Lionel Andrés Messi Cuccittini         |         12.4 |         53.6 |                   2.3 |                  43.3 | Saved          |
## |  6 | Barcelona |       33 | Arthur Henrique Ramos de Oliveira Melo |         11   |         27.1 |                   8.8 |                  30.2 | Saved          |
## |  7 | Barcelona |       36 | Antoine Griezmann                      |         19.2 |         37.4 |                  18.1 |                  37.7 | Blocked        |
## |  9 | Barcelona |       37 | Lionel Andrés Messi Cuccittini         |         17.3 |         40.7 |                   4.9 |                  40   | Saved          |
## | 10 | Barcelona |       39 | Arturo Erasmo Vidal Pardo              |         17.4 |         42.7 |                  16.2 |                  42.5 | Blocked        |
## | 11 | Barcelona |       46 | Lionel Andrés Messi Cuccittini         |         17.2 |         26.2 |                  16.3 |                  26.9 | Blocked        |
## | 17 | Barcelona |       69 | Martin Braithwaite Christensen         |          8.9 |         57.5 |                   7.6 |                  54.5 | Saved          |
```

Once we are satisfied, we will finally visualize the shot map and the heat map:


```python
# Pitch drawing code
pitch = Pitch(pitch_color = 'black', line_color = 'white', constrained_layout = True, tight_layout = False, goal_type = 'box')
fig, ax = pitch.draw()

# Heat map code
res = sns.kdeplot(events_shot_Barca['location_x'], events_shot_Barca['location_y'], fill = True, 
thresh = 0.05, alpha = 0.5, levels = 10, cmap = 'Purples_d')

# Pass map code
```

```python
pitch.arrows(events_shot_Barca_off.location_x, events_shot_Barca_off.location_y, events_shot_Barca_off.shot_end_location_x, events_shot_Barca_off.shot_end_location_y, ax=ax, color='red', width = 3, label = 'Off T')
```

```python
pitch.scatter(events_shot_Barca_off.location_x, events_shot_Barca_off.location_y, ax = ax, color = 'red')
```

```python
pitch.arrows(events_shot_Barca_saved_blocked.location_x, events_shot_Barca_saved_blocked.location_y, events_shot_Barca_saved_blocked.shot_end_location_x, events_shot_Barca_saved_blocked.shot_end_location_y, ax=ax, color='orange', width = 3, label = 'Blocked / Saved')
```

```python
pitch.scatter(events_shot_Barca_saved_blocked.location_x, events_shot_Barca_saved_blocked.location_y, ax = ax, color = 'orange')

# General plot code
```

```python
ax.legend(handlelength = 3, edgecolor='None', fontsize = 10)
```

```python
plt.title("Barcelona shot and heat map")
```

```python
plt.show()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-11.png" width="672" />

Notice that some of the shots from `Barcelona` were blocked very close to where the shots were taken from. The lengths of the arrows for these cases are compact. Whoo! We have come to the end of this tutorial. I know this tutorial was one hell of a ride. But I can guarantee that if the reader goes through it thoroughly, they can grasp a lot of useful information on soccer data analysis. In the next post tutorial, we will study how to build and visualize pass network of a particular team during a particular match and how to use concepts of *complex network theory* to analyze these pass networks with the help of `NetworkX` Python package. 


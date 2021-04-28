---
title: Pass Network Analysis
author: Indranil Ghosh
date: '2021-04-28'
slug: pass-network-analysis
categories: ["Python", "visualization", "NetworkX"]
tags: ["statsbomb api", "NetworkX", "Network Analysis", "Pass Network"]
subtitle: ''
summary: ''
authors: []
lastmod: '2021-04-28T10:15:58+05:30'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

In our [last tutorial](https://realsoccerexpand.netlify.app/post/pass-map-shot-map-and-heat-map/) we studied how to draw a pass map, a shot map and their corresponding heat maps. We used statsbomb's open even data from the match between *Real Madrid* and *Barcelona*, which *Real Madrid* ended up winning 2-0. In this post we will again use statsbomb's open event passing data (from a separate game this time, which we will decide on the go) and visualize the resulting pass network of a particular team on the football pitch. We will then use basic concepts from complex network analysis literature to further analyze the network and deduce some results. We will employ the [`NetworkX`](https://networkx.org/) Python package for the analysis purpose. 

Let us `pip` install the package:


```python
pip install networkx
```

After installing the package we will import all the necessary packages and modules:


```python
from statsbombpy import sb # statsbomb api
import matplotlib.pyplot as plt # matplotlib for plotting
from mplsoccer.pitch import Pitch # for drawing the football pitch
import seaborn as sns # seaborn for plotting useful statistical graphs
import numpy as np # numerical python package
import pandas as pd # pandas for manipulating and analysing data
import networkx as nx # package for complex network analysis
```

Let us again work step by step to fetch the event data from a particular match:


```python
comp = sb.competitions()
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
## | 15 |               37 |          42 | England                  | FA Women's Super League | female               | 2019/2020     | 2021-04-28T11:39:57.090    | 2021-04-27T06:27:03.599355 |
## | 16 |               37 |           4 | England                  | FA Women's Super League | female               | 2018/2019     | 2021-04-28T11:09:58.484    | 2021-04-28T07:08:31.988445 |
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

Let us use the first row from `comp` where the `competition_id` is `16` and `season_id` is `4`. We see that it holds the event data from *UEFA Champions League* ⚽ for the 2017-18 season. Let us now extract out the matches using the above information.


```python
mat = sb.matches(competition_id = 16, season_id = 1)
```


```python
print(mat.to_markdown())
```

```
## |    |   match_id | match_date   | kick_off     | competition               | season    | home_team   | away_team   |   home_score |   away_score | match_status   | match_status_360   | last_updated               | last_updated_360   |   match_week | competition_stage   | stadium          | referee   | data_version   |   shot_fidelity_version |   xy_fidelity_version |
## |---:|-----------:|:-------------|:-------------|:--------------------------|:----------|:------------|:------------|-------------:|-------------:|:---------------|:-------------------|:---------------------------|:-------------------|-------------:|:--------------------|:-----------------|:----------|:---------------|------------------------:|----------------------:|
## |  0 |      18245 | 2018-05-26   | 20:45:00.000 | Europe - Champions League | 2017/2018 | Real Madrid | Liverpool   |            3 |            1 | available      | unscheduled        | 2021-01-23T21:55:30.425330 |                    |            7 | Final               | NSK Olimpijs'kyj | M. Mažić  | 1.1.0          |                       2 |                     2 |
```

We see there is only one match available, having `match_id` set to `18245`. It represents the final that took place between *Real Madrid* and *Liverpool* at *Olimpiyskiy National Sports Complex, Moscow* and Real Madrid won with the full time score: 3-1. Finally let us draw out the complete event data from this match:


```python
events = sb.events(match_id = 18245)
```

We will print the first and the last 10 rows of the dataset to get an idea of how it looks and what information it provides us with:


```python
print(events.head(10).to_markdown())
```

```
## |    |   50_50 |   ball_receipt_outcome |   ball_recovery_recovery_failure |   block_offensive |   carry_end_location |   clearance_aerial_won |   clearance_body_part |   clearance_head |   clearance_left_foot |   clearance_right_foot |   counterpress |   dribble_nutmeg |   dribble_outcome |   dribble_overrun |   duel_outcome |   duel_type |   duration |   foul_committed_advantage |   foul_committed_card |   foul_committed_type |   foul_won_advantage |   foul_won_defensive |   goalkeeper_body_part |   goalkeeper_end_location |   goalkeeper_outcome |   goalkeeper_position |   goalkeeper_punched_out |   goalkeeper_technique |   goalkeeper_type | id                                   |   index |   injury_stoppage_in_chain |   interception_outcome | location     |   match_id |   minute |   off_camera |   out |   pass_aerial_won |   pass_angle |   pass_assisted_shot_id | pass_body_part   |   pass_cross |   pass_cut_back | pass_end_location   |   pass_goal_assist | pass_height   |   pass_inswinging |   pass_length |   pass_miscommunication | pass_outcome   |   pass_outswinging | pass_recipient                      |   pass_shot_assist |   pass_straight |   pass_switch |   pass_technique |   pass_through_ball | pass_type   |   period | play_pattern   | player              | position              |   possession | possession_team   | related_events                                                                                                           |   second |   shot_aerial_won |   shot_body_part |   shot_end_location |   shot_first_time |   shot_freeze_frame |   shot_key_pass_id |   shot_one_on_one |   shot_outcome |   shot_redirect |   shot_statsbomb_xg |   shot_technique |   shot_type |   substitution_outcome |   substitution_replacement | tactics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | team        | timestamp    | type        |   under_pressure |
## |---:|--------:|-----------------------:|---------------------------------:|------------------:|---------------------:|-----------------------:|----------------------:|-----------------:|----------------------:|-----------------------:|---------------:|-----------------:|------------------:|------------------:|---------------:|------------:|-----------:|---------------------------:|----------------------:|----------------------:|---------------------:|---------------------:|-----------------------:|--------------------------:|---------------------:|----------------------:|-------------------------:|-----------------------:|------------------:|:-------------------------------------|--------:|---------------------------:|-----------------------:|:-------------|-----------:|---------:|-------------:|------:|------------------:|-------------:|------------------------:|:-----------------|-------------:|----------------:|:--------------------|-------------------:|:--------------|------------------:|--------------:|------------------------:|:---------------|-------------------:|:------------------------------------|-------------------:|----------------:|--------------:|-----------------:|--------------------:|:------------|---------:|:---------------|:--------------------|:----------------------|-------------:|:------------------|:-------------------------------------------------------------------------------------------------------------------------|---------:|------------------:|-----------------:|--------------------:|------------------:|--------------------:|-------------------:|------------------:|---------------:|----------------:|--------------------:|-----------------:|------------:|-----------------------:|---------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------|:-------------|:------------|-----------------:|
## |  0 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |   0        |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | 5eee3ffd-f0c0-4532-868b-4a66cbf20cb8 |       1 |                        nan |                    nan | nan          |      18245 |        0 |          nan |   nan |               nan |   nan        |                     nan | nan              |          nan |             nan | nan                 |                nan | nan           |               nan |      nan      |                     nan | nan            |                nan | nan                                 |                nan |             nan |           nan |              nan |                 nan | nan         |        1 | Regular Play   | nan                 | nan                   |            1 | Real Madrid       | nan                                                                                                                      |        0 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | {'formation': 41212, 'lineup': [{'player': {'id': 5597, 'name': 'Keylor Navas Gamboa'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 5721, 'name': 'Daniel Carvajal Ramos'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 2}, {'player': {'id': 5485, 'name': 'Raphaël Varane'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 5}, {'player': {'id': 5201, 'name': 'Sergio Ramos García'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 5552, 'name': 'Marcelo Vieira da Silva Júnior'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 12}, {'player': {'id': 5539, 'name': 'Carlos Henrique Casimiro'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 5463, 'name': 'Luka Modrić'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 10}, {'player': {'id': 5574, 'name': 'Toni Kroos'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 8}, {'player': {'id': 4926, 'name': 'Francisco Román Alarcón Suárez'}, 'position': {'id': 19, 'name': 'Center Attacking Midfield'}, 'jersey_number': 22}, {'player': {'id': 19677, 'name': 'Karim Benzema'}, 'position': {'id': 22, 'name': 'Right Center Forward'}, 'jersey_number': 9}, {'player': {'id': 5207, 'name': 'Cristiano Ronaldo dos Santos Aveiro'}, 'position': {'id': 24, 'name': 'Left Center Forward'}, 'jersey_number': 7}]} | Real Madrid | 00:00:00.000 | Starting XI |              nan |
## |  1 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |   0        |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | eaa65a92-02d3-4375-b2b7-7c2f679a620c |       2 |                        nan |                    nan | nan          |      18245 |        0 |          nan |   nan |               nan |   nan        |                     nan | nan              |          nan |             nan | nan                 |                nan | nan           |               nan |      nan      |                     nan | nan            |                nan | nan                                 |                nan |             nan |           nan |              nan |                 nan | nan         |        1 | Regular Play   | nan                 | nan                   |            1 | Real Madrid       | nan                                                                                                                      |        0 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | {'formation': 433, 'lineup': [{'player': {'id': 3630, 'name': 'Loris Karius'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 3664, 'name': 'Trent Alexander-Arnold'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 66}, {'player': {'id': 3471, 'name': 'Dejan Lovren'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 6}, {'player': {'id': 3669, 'name': 'Virgil van Dijk'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 3655, 'name': 'Andrew Robertson'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 26}, {'player': {'id': 3532, 'name': 'Jordan Brian Henderson'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 3567, 'name': 'Georginio Wijnaldum'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 5}, {'player': {'id': 3473, 'name': 'James Philip Milner'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 7}, {'player': {'id': 3531, 'name': 'Mohamed Salah'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'jersey_number': 11}, {'player': {'id': 3629, 'name': 'Sadio Mané'}, 'position': {'id': 21, 'name': 'Left Wing'}, 'jersey_number': 19}, {'player': {'id': 3535, 'name': 'Roberto Firmino Barbosa de Oliveira'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'jersey_number': 9}]}                                                                 | Liverpool   | 00:00:00.000 | Starting XI |              nan |
## |  2 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |   0        |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | 9c82d2e5-ebba-4825-b7f9-b11b04433ed8 |       3 |                        nan |                    nan | nan          |      18245 |        0 |          nan |   nan |               nan |   nan        |                     nan | nan              |          nan |             nan | nan                 |                nan | nan           |               nan |      nan      |                     nan | nan            |                nan | nan                                 |                nan |             nan |           nan |              nan |                 nan | nan         |        1 | Regular Play   | nan                 | nan                   |            1 | Real Madrid       | ['b791047a-3eea-452f-b3a9-212bd40cd7cb']                                                                                 |        0 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Real Madrid | 00:00:00.000 | Half Start  |              nan |
## |  3 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |   0        |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | b791047a-3eea-452f-b3a9-212bd40cd7cb |       4 |                        nan |                    nan | nan          |      18245 |        0 |          nan |   nan |               nan |   nan        |                     nan | nan              |          nan |             nan | nan                 |                nan | nan           |               nan |      nan      |                     nan | nan            |                nan | nan                                 |                nan |             nan |           nan |              nan |                 nan | nan         |        1 | Regular Play   | nan                 | nan                   |            1 | Real Madrid       | ['9c82d2e5-ebba-4825-b7f9-b11b04433ed8']                                                                                 |        0 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Liverpool   | 00:00:00.000 | Half Start  |              nan |
## |  4 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |   0        |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | c6dbfa9c-faf1-4f33-986e-357706c33dd0 |    1924 |                        nan |                    nan | nan          |      18245 |       45 |          nan |   nan |               nan |   nan        |                     nan | nan              |          nan |             nan | nan                 |                nan | nan           |               nan |      nan      |                     nan | nan            |                nan | nan                                 |                nan |             nan |           nan |              nan |                 nan | nan         |        2 | From Goal Kick | nan                 | nan                   |           85 | Liverpool         | ['85d8a8b2-022f-493b-a161-cb6baec082a5']                                                                                 |        0 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Liverpool   | 00:00:00.000 | Half Start  |              nan |
## |  5 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |   0        |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | 85d8a8b2-022f-493b-a161-cb6baec082a5 |    1925 |                        nan |                    nan | nan          |      18245 |       45 |          nan |   nan |               nan |   nan        |                     nan | nan              |          nan |             nan | nan                 |                nan | nan           |               nan |      nan      |                     nan | nan            |                nan | nan                                 |                nan |             nan |           nan |              nan |                 nan | nan         |        2 | From Goal Kick | nan                 | nan                   |           85 | Liverpool         | ['c6dbfa9c-faf1-4f33-986e-357706c33dd0']                                                                                 |        0 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Real Madrid | 00:00:00.000 | Half Start  |              nan |
## |  6 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |   1.73591  |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | 25be91a5-a084-42cb-8cc1-a0fe7b0f52f9 |       5 |                        nan |                    nan | [60.0, 40.0] |      18245 |        0 |          nan |   nan |               nan |     3.09861  |                     nan | Right Foot       |          nan |             nan | [32.1, 41.2]        |                nan | Ground Pass   |               nan |       27.9258 |                     nan | nan            |                nan | Dejan Lovren                        |                nan |             nan |           nan |              nan |                 nan | Kick Off    |        1 | From Kick Off  | James Philip Milner | Left Center Midfield  |            2 | Liverpool         | ['e1a3ac58-89f3-42c7-aacc-1592ad5ab8f3']                                                                                 |        0 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Liverpool   | 00:00:00.371 | Pass        |              nan |
## |  7 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |   3.77207  |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | b544eb38-7cc9-4cb5-99e9-ebac4fce9eb9 |       8 |                        nan |                    nan | [35.0, 40.8] |      18245 |        0 |          nan |   nan |               nan |    -0.30397  |                     nan | Right Foot       |          nan |             nan | [92.7, 22.7]        |                nan | High Pass     |               nan |       60.4723 |                     nan | Incomplete     |                nan | Roberto Firmino Barbosa de Oliveira |                nan |             nan |           nan |              nan |                 nan | nan         |        1 | From Kick Off  | Dejan Lovren        | Right Center Back     |            2 | Liverpool         | ['ad23249c-f7da-4e2d-aa13-8a13df1afc57', 'ae5939bf-8467-4dfa-95df-ebba7c70d4c8']                                         |        3 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Liverpool   | 00:00:03.275 | Pass        |              nan |
## |  8 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |   0.793057 |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | 192c9629-1703-40ab-8498-cab869cf0144 |      12 |                        nan |                    nan | [27.4, 60.2] |      18245 |        0 |          nan |   nan |               nan |     0.918927 |                     nan | Right Foot       |          nan |             nan | [36.1, 71.6]        |                nan | High Pass     |               nan |       14.3405 |                     nan | nan            |                nan | Luka Modrić                         |                nan |             nan |           nan |              nan |                 nan | nan         |        1 | Regular Play   | Raphaël Varane      | Right Center Back     |            3 | Real Madrid       | ['a88801db-695b-4a7e-ac98-48a2e1b8b61c']                                                                                 |        8 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Real Madrid | 00:00:08.236 | Pass        |              nan |
## |  9 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |   0.987721 |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | 599fb72e-8baf-4d5b-815e-7beb8c54e6af |      17 |                        nan |                    nan | [35.3, 75.4] |      18245 |        0 |          nan |   nan |               nan |     3.04884  |                     nan | Right Foot       |          nan |             nan | [22.4, 76.6]        |                nan | Low Pass      |               nan |       12.9557 |                     nan | nan            |                nan | Daniel Carvajal Ramos               |                nan |             nan |           nan |              nan |                 nan | nan         |        1 | Regular Play   | Luka Modrić         | Right Center Midfield |            3 | Real Madrid       | ['5016d9e2-20f6-4d4b-a19f-3c7d5999c8d4', '77afbacb-6fe9-47a2-857b-98c47a785e6a', 'b9d35239-8e68-4d04-a74d-ec653114de99'] |       10 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan |                    nan |                        nan | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Real Madrid | 00:00:10.701 | Pass        |                1 |
```


```python
print(events.tail(10).to_markdown())
```

```
## |      |   50_50 |   ball_receipt_outcome |   ball_recovery_recovery_failure |   block_offensive |   carry_end_location |   clearance_aerial_won |   clearance_body_part |   clearance_head |   clearance_left_foot |   clearance_right_foot |   counterpress |   dribble_nutmeg |   dribble_outcome |   dribble_overrun |   duel_outcome |   duel_type |   duration |   foul_committed_advantage |   foul_committed_card |   foul_committed_type |   foul_won_advantage |   foul_won_defensive |   goalkeeper_body_part |   goalkeeper_end_location |   goalkeeper_outcome |   goalkeeper_position |   goalkeeper_punched_out |   goalkeeper_technique |   goalkeeper_type | id                                   |   index |   injury_stoppage_in_chain |   interception_outcome | location      |   match_id |   minute |   off_camera |   out |   pass_aerial_won |   pass_angle |   pass_assisted_shot_id |   pass_body_part |   pass_cross |   pass_cut_back |   pass_end_location |   pass_goal_assist |   pass_height |   pass_inswinging |   pass_length |   pass_miscommunication |   pass_outcome |   pass_outswinging |   pass_recipient |   pass_shot_assist |   pass_straight |   pass_switch |   pass_technique |   pass_through_ball |   pass_type |   period | play_pattern   | player              | position             |   possession | possession_team   | related_events                           |   second |   shot_aerial_won |   shot_body_part |   shot_end_location |   shot_first_time |   shot_freeze_frame |   shot_key_pass_id |   shot_one_on_one |   shot_outcome |   shot_redirect |   shot_statsbomb_xg |   shot_technique |   shot_type | substitution_outcome   | substitution_replacement   | tactics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | team        | timestamp    | type           |   under_pressure |
## |-----:|--------:|-----------------------:|---------------------------------:|------------------:|---------------------:|-----------------------:|----------------------:|-----------------:|----------------------:|-----------------------:|---------------:|-----------------:|------------------:|------------------:|---------------:|------------:|-----------:|---------------------------:|----------------------:|----------------------:|---------------------:|---------------------:|-----------------------:|--------------------------:|---------------------:|----------------------:|-------------------------:|-----------------------:|------------------:|:-------------------------------------|--------:|---------------------------:|-----------------------:|:--------------|-----------:|---------:|-------------:|------:|------------------:|-------------:|------------------------:|-----------------:|-------------:|----------------:|--------------------:|-------------------:|--------------:|------------------:|--------------:|------------------------:|---------------:|-------------------:|-----------------:|-------------------:|----------------:|--------------:|-----------------:|--------------------:|------------:|---------:|:---------------|:--------------------|:---------------------|-------------:|:------------------|:-----------------------------------------|---------:|------------------:|-----------------:|--------------------:|------------------:|--------------------:|-------------------:|------------------:|---------------:|----------------:|--------------------:|-----------------:|------------:|:-----------------------|:---------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------|:-------------|:---------------|-----------------:|
## | 3487 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | 8fafbd4c-f4e0-4bc4-a518-9a27dee8c354 |    3199 |                        nan |                    nan | nan           |      18245 |       82 |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |                 nan |                nan |           nan |               nan |           nan |                     nan |            nan |                nan |              nan |                nan |             nan |           nan |              nan |                 nan |         nan |        2 | From Throw In  | James Philip Milner | Left Center Midfield |          146 | Real Madrid       | nan                                      |       27 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | Tactical               | Emre Can                   | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Liverpool   | 00:37:27.766 | Substitution   |              nan |
## | 3488 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | bcb01505-642d-4b35-9fb6-63bec1bf79d9 |    3349 |                        nan |                    nan | nan           |      18245 |       88 |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |                 nan |                nan |           nan |               nan |           nan |                     nan |            nan |                nan |              nan |                nan |             nan |           nan |              nan |                 nan |         nan |        2 | From Free Kick | Karim Benzema       | Center Forward       |          154 | Liverpool         | nan                                      |       21 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | Tactical               | Marco Asensio Willemsen    | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Real Madrid | 00:43:21.957 | Substitution   |              nan |
## | 3489 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | 7062c012-6d9c-4d89-833c-b91c427676ed |    1278 |                        nan |                    nan | nan           |      18245 |       31 |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |                 nan |                nan |           nan |               nan |           nan |                     nan |            nan |                nan |              nan |                nan |             nan |           nan |              nan |                 nan |         nan |        1 | From Throw In  | nan                 | nan                  |           60 | Real Madrid       | nan                                      |       41 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | nan                    | nan                        | {'formation': 433, 'lineup': [{'player': {'id': 3630, 'name': 'Loris Karius'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 3664, 'name': 'Trent Alexander-Arnold'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 66}, {'player': {'id': 3471, 'name': 'Dejan Lovren'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 6}, {'player': {'id': 3669, 'name': 'Virgil van Dijk'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 3655, 'name': 'Andrew Robertson'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 26}, {'player': {'id': 3532, 'name': 'Jordan Brian Henderson'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 3567, 'name': 'Georginio Wijnaldum'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 5}, {'player': {'id': 3473, 'name': 'James Philip Milner'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 7}, {'player': {'id': 3629, 'name': 'Sadio Mané'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'jersey_number': 19}, {'player': {'id': 4090, 'name': 'Adam David Lallana'}, 'position': {'id': 21, 'name': 'Left Wing'}, 'jersey_number': 20}, {'player': {'id': 3535, 'name': 'Roberto Firmino Barbosa de Oliveira'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'jersey_number': 9}]}                                  | Liverpool   | 00:31:41.916 | Tactical Shift |              nan |
## | 3490 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | c659bbc3-5700-4fcb-85ad-b0e13e42023f |    2431 |                        nan |                    nan | nan           |      18245 |       61 |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |                 nan |                nan |           nan |               nan |           nan |                     nan |            nan |                nan |              nan |                nan |             nan |           nan |              nan |                 nan |         nan |        2 | From Corner    | nan                 | nan                  |          110 | Real Madrid       | nan                                      |        1 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | nan                    | nan                        | {'formation': 433, 'lineup': [{'player': {'id': 5597, 'name': 'Keylor Navas Gamboa'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 5202, 'name': 'José Ignacio Fernández Iglesias'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 6}, {'player': {'id': 5485, 'name': 'Raphaël Varane'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 5}, {'player': {'id': 5201, 'name': 'Sergio Ramos García'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 5552, 'name': 'Marcelo Vieira da Silva Júnior'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 12}, {'player': {'id': 5539, 'name': 'Carlos Henrique Casimiro'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 5463, 'name': 'Luka Modrić'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 10}, {'player': {'id': 5574, 'name': 'Toni Kroos'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 8}, {'player': {'id': 6399, 'name': 'Gareth Frank Bale'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'jersey_number': 11}, {'player': {'id': 5207, 'name': 'Cristiano Ronaldo dos Santos Aveiro'}, 'position': {'id': 21, 'name': 'Left Wing'}, 'jersey_number': 7}, {'player': {'id': 19677, 'name': 'Karim Benzema'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'jersey_number': 9}]}           | Real Madrid | 00:16:01.724 | Tactical Shift |              nan |
## | 3491 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | 5576a614-3af2-4fc0-8636-151a4508a1d7 |    3350 |                        nan |                    nan | nan           |      18245 |       88 |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |                 nan |                nan |           nan |               nan |           nan |                     nan |            nan |                nan |              nan |                nan |             nan |           nan |              nan |                 nan |         nan |        2 | From Free Kick | nan                 | nan                  |          154 | Liverpool         | nan                                      |       34 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | nan                    | nan                        | {'formation': 433, 'lineup': [{'player': {'id': 5597, 'name': 'Keylor Navas Gamboa'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 5202, 'name': 'José Ignacio Fernández Iglesias'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 6}, {'player': {'id': 5485, 'name': 'Raphaël Varane'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 5}, {'player': {'id': 5201, 'name': 'Sergio Ramos García'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 5552, 'name': 'Marcelo Vieira da Silva Júnior'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 12}, {'player': {'id': 5539, 'name': 'Carlos Henrique Casimiro'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 5463, 'name': 'Luka Modrić'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 10}, {'player': {'id': 5574, 'name': 'Toni Kroos'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 8}, {'player': {'id': 6399, 'name': 'Gareth Frank Bale'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'jersey_number': 11}, {'player': {'id': 5719, 'name': 'Marco Asensio Willemsen'}, 'position': {'id': 21, 'name': 'Left Wing'}, 'jersey_number': 20}, {'player': {'id': 5207, 'name': 'Cristiano Ronaldo dos Santos Aveiro'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'jersey_number': 7}]} | Real Madrid | 00:43:34.632 | Tactical Shift |              nan |
## | 3492 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | 6bc1c5c3-86ff-4ecb-882c-5d0ecb24a654 |    1730 |                        nan |                    nan | [114.8, 41.4] |      18245 |       42 |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |                 nan |                nan |           nan |               nan |           nan |                     nan |            nan |                nan |              nan |                nan |             nan |           nan |              nan |                 nan |         nan |        1 | Regular Play   | Karim Benzema       | Right Center Forward |           73 | Real Madrid       | nan                                      |       21 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | nan                    | nan                        | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Real Madrid | 00:42:21.211 | Offside        |              nan |
## | 3493 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | cfb22dca-62c8-4c66-bee7-46c52cfb25fa |    1922 |                        nan |                    nan | nan           |      18245 |       48 |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |                 nan |                nan |           nan |               nan |           nan |                     nan |            nan |                nan |              nan |                nan |             nan |           nan |              nan |                 nan |         nan |        1 | From Goal Kick | nan                 | nan                  |           85 | Liverpool         | ['3344a029-5f88-48de-b70c-925c4363dcd3'] |       31 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | nan                    | nan                        | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Real Madrid | 00:48:31.725 | Half End       |              nan |
## | 3494 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | 3344a029-5f88-48de-b70c-925c4363dcd3 |    1923 |                        nan |                    nan | nan           |      18245 |       48 |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |                 nan |                nan |           nan |               nan |           nan |                     nan |            nan |                nan |              nan |                nan |             nan |           nan |              nan |                 nan |         nan |        1 | From Goal Kick | nan                 | nan                  |           85 | Liverpool         | ['cfb22dca-62c8-4c66-bee7-46c52cfb25fa'] |       31 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | nan                    | nan                        | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Liverpool   | 00:48:31.725 | Half End       |              nan |
## | 3495 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | ce7d446a-e8bf-4631-bcf5-2bd323ba251e |    3496 |                        nan |                    nan | nan           |      18245 |       93 |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |                 nan |                nan |           nan |               nan |           nan |                     nan |            nan |                nan |              nan |                nan |             nan |           nan |              nan |                 nan |         nan |        2 | Regular Play   | nan                 | nan                  |          164 | Real Madrid       | ['d19b2348-de55-4bbf-9b1f-e44d95aa3a77'] |        2 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | nan                    | nan                        | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Liverpool   | 00:48:02.893 | Half End       |              nan |
## | 3496 |     nan |                    nan |                              nan |               nan |                  nan |                    nan |                   nan |              nan |                   nan |                    nan |            nan |              nan |               nan |               nan |            nan |         nan |          0 |                        nan |                   nan |                   nan |                  nan |                  nan |                    nan |                       nan |                  nan |                   nan |                      nan |                    nan |               nan | d19b2348-de55-4bbf-9b1f-e44d95aa3a77 |    3497 |                        nan |                    nan | nan           |      18245 |       93 |          nan |   nan |               nan |          nan |                     nan |              nan |          nan |             nan |                 nan |                nan |           nan |               nan |           nan |                     nan |            nan |                nan |              nan |                nan |             nan |           nan |              nan |                 nan |         nan |        2 | Regular Play   | nan                 | nan                  |          164 | Real Madrid       | ['ce7d446a-e8bf-4631-bcf5-2bd323ba251e'] |        2 |               nan |              nan |                 nan |               nan |                 nan |                nan |               nan |            nan |             nan |                 nan |              nan |         nan | nan                    | nan                        | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Real Madrid | 00:48:02.893 | Half End       |              nan |
```

As we have been usually doing till now, let us print out the column names of `events` to get an overview of the relevant and the unnecessary rows for this tutorial.


```python
print(events.columns)
```

```
## Index(['50_50', 'ball_receipt_outcome', 'ball_recovery_recovery_failure',
##        'block_offensive', 'carry_end_location', 'clearance_aerial_won',
##        'clearance_body_part', 'clearance_head', 'clearance_left_foot',
##        'clearance_right_foot', 'counterpress', 'dribble_nutmeg',
##        'dribble_outcome', 'dribble_overrun', 'duel_outcome', 'duel_type',
##        'duration', 'foul_committed_advantage', 'foul_committed_card',
##        'foul_committed_type', 'foul_won_advantage', 'foul_won_defensive',
##        'goalkeeper_body_part', 'goalkeeper_end_location', 'goalkeeper_outcome',
##        'goalkeeper_position', 'goalkeeper_punched_out', 'goalkeeper_technique',
##        'goalkeeper_type', 'id', 'index', 'injury_stoppage_in_chain',
##        'interception_outcome', 'location', 'match_id', 'minute', 'off_camera',
##        'out', 'pass_aerial_won', 'pass_angle', 'pass_assisted_shot_id',
##        'pass_body_part', 'pass_cross', 'pass_cut_back', 'pass_end_location',
##        'pass_goal_assist', 'pass_height', 'pass_inswinging', 'pass_length',
##        'pass_miscommunication', 'pass_outcome', 'pass_outswinging',
##        'pass_recipient', 'pass_shot_assist', 'pass_straight', 'pass_switch',
##        'pass_technique', 'pass_through_ball', 'pass_type', 'period',
##        'play_pattern', 'player', 'position', 'possession', 'possession_team',
##        'related_events', 'second', 'shot_aerial_won', 'shot_body_part',
##        'shot_end_location', 'shot_first_time', 'shot_freeze_frame',
##        'shot_key_pass_id', 'shot_one_on_one', 'shot_outcome', 'shot_redirect',
##        'shot_statsbomb_xg', 'shot_technique', 'shot_type',
##        'substitution_outcome', 'substitution_replacement', 'tactics', 'team',
##        'timestamp', 'type', 'under_pressure'],
##       dtype='object')
```

If we look into the `events` dataset, we notice that the `tactics` column provides us with team lineups, formations, player ids and their jersey number from both the teams. The corresponding row values for column `type` gives us an idea about whether it was the starting 11 formation or was a tactical shift or any other developments in the teams. Let us generate a completely new dataset only focusing on the `tactics` and the `type` columns. We will filter the data in such a way that the `tactics` column has no rows set to `nan`.


```python
tact = events[events['tactics'].isnull() == False]
tact = tact[['tactics', 'team', 'type']]
print(tact.to_markdown())
```

```
## |      | tactics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | team        | type           |
## |-----:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------|:---------------|
## |    0 | {'formation': 41212, 'lineup': [{'player': {'id': 5597, 'name': 'Keylor Navas Gamboa'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 5721, 'name': 'Daniel Carvajal Ramos'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 2}, {'player': {'id': 5485, 'name': 'Raphaël Varane'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 5}, {'player': {'id': 5201, 'name': 'Sergio Ramos García'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 5552, 'name': 'Marcelo Vieira da Silva Júnior'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 12}, {'player': {'id': 5539, 'name': 'Carlos Henrique Casimiro'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 5463, 'name': 'Luka Modrić'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 10}, {'player': {'id': 5574, 'name': 'Toni Kroos'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 8}, {'player': {'id': 4926, 'name': 'Francisco Román Alarcón Suárez'}, 'position': {'id': 19, 'name': 'Center Attacking Midfield'}, 'jersey_number': 22}, {'player': {'id': 19677, 'name': 'Karim Benzema'}, 'position': {'id': 22, 'name': 'Right Center Forward'}, 'jersey_number': 9}, {'player': {'id': 5207, 'name': 'Cristiano Ronaldo dos Santos Aveiro'}, 'position': {'id': 24, 'name': 'Left Center Forward'}, 'jersey_number': 7}]} | Real Madrid | Starting XI    |
## |    1 | {'formation': 433, 'lineup': [{'player': {'id': 3630, 'name': 'Loris Karius'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 3664, 'name': 'Trent Alexander-Arnold'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 66}, {'player': {'id': 3471, 'name': 'Dejan Lovren'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 6}, {'player': {'id': 3669, 'name': 'Virgil van Dijk'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 3655, 'name': 'Andrew Robertson'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 26}, {'player': {'id': 3532, 'name': 'Jordan Brian Henderson'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 3567, 'name': 'Georginio Wijnaldum'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 5}, {'player': {'id': 3473, 'name': 'James Philip Milner'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 7}, {'player': {'id': 3531, 'name': 'Mohamed Salah'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'jersey_number': 11}, {'player': {'id': 3629, 'name': 'Sadio Mané'}, 'position': {'id': 21, 'name': 'Left Wing'}, 'jersey_number': 19}, {'player': {'id': 3535, 'name': 'Roberto Firmino Barbosa de Oliveira'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'jersey_number': 9}]}                                                                 | Liverpool   | Starting XI    |
## | 3489 | {'formation': 433, 'lineup': [{'player': {'id': 3630, 'name': 'Loris Karius'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 3664, 'name': 'Trent Alexander-Arnold'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 66}, {'player': {'id': 3471, 'name': 'Dejan Lovren'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 6}, {'player': {'id': 3669, 'name': 'Virgil van Dijk'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 3655, 'name': 'Andrew Robertson'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 26}, {'player': {'id': 3532, 'name': 'Jordan Brian Henderson'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 3567, 'name': 'Georginio Wijnaldum'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 5}, {'player': {'id': 3473, 'name': 'James Philip Milner'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 7}, {'player': {'id': 3629, 'name': 'Sadio Mané'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'jersey_number': 19}, {'player': {'id': 4090, 'name': 'Adam David Lallana'}, 'position': {'id': 21, 'name': 'Left Wing'}, 'jersey_number': 20}, {'player': {'id': 3535, 'name': 'Roberto Firmino Barbosa de Oliveira'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'jersey_number': 9}]}                                                            | Liverpool   | Tactical Shift |
## | 3490 | {'formation': 433, 'lineup': [{'player': {'id': 5597, 'name': 'Keylor Navas Gamboa'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 5202, 'name': 'José Ignacio Fernández Iglesias'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 6}, {'player': {'id': 5485, 'name': 'Raphaël Varane'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 5}, {'player': {'id': 5201, 'name': 'Sergio Ramos García'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 5552, 'name': 'Marcelo Vieira da Silva Júnior'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 12}, {'player': {'id': 5539, 'name': 'Carlos Henrique Casimiro'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 5463, 'name': 'Luka Modrić'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 10}, {'player': {'id': 5574, 'name': 'Toni Kroos'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 8}, {'player': {'id': 6399, 'name': 'Gareth Frank Bale'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'jersey_number': 11}, {'player': {'id': 5207, 'name': 'Cristiano Ronaldo dos Santos Aveiro'}, 'position': {'id': 21, 'name': 'Left Wing'}, 'jersey_number': 7}, {'player': {'id': 19677, 'name': 'Karim Benzema'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'jersey_number': 9}]}                                     | Real Madrid | Tactical Shift |
## | 3491 | {'formation': 433, 'lineup': [{'player': {'id': 5597, 'name': 'Keylor Navas Gamboa'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 5202, 'name': 'José Ignacio Fernández Iglesias'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 6}, {'player': {'id': 5485, 'name': 'Raphaël Varane'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 5}, {'player': {'id': 5201, 'name': 'Sergio Ramos García'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 5552, 'name': 'Marcelo Vieira da Silva Júnior'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 12}, {'player': {'id': 5539, 'name': 'Carlos Henrique Casimiro'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 5463, 'name': 'Luka Modrić'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 10}, {'player': {'id': 5574, 'name': 'Toni Kroos'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 8}, {'player': {'id': 6399, 'name': 'Gareth Frank Bale'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'jersey_number': 11}, {'player': {'id': 5719, 'name': 'Marco Asensio Willemsen'}, 'position': {'id': 21, 'name': 'Left Wing'}, 'jersey_number': 20}, {'player': {'id': 5207, 'name': 'Cristiano Ronaldo dos Santos Aveiro'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'jersey_number': 7}]}                           | Real Madrid | Tactical Shift |
```

Let us focus only on the tactics for the starting 11 set up from both the teams. We will build and analyze the pass network generated from among the starting 11 players from either of the teams. If we look into the first two rows of the `type` column in `tact`, we see that they are set as `'Starting XI'`, one for each team. Let us separately fetch the data for the teams, filtering by `type`


```python
tact = tact[tact['type'] == 'Starting XI']
tact_Real = tact[tact['team'] == 'Real Madrid']
tact_Liv = tact[tact['team'] == 'Liverpool']
tact_Real = tact_Real['tactics']
tact_Liv = tact_Liv['tactics']
```

Let us see how `tact_Real` and `tact_Barca` look:


```python
print(tact_Real.to_markdown())
```

```
## |    | tactics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
## |---:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
## |  0 | {'formation': 41212, 'lineup': [{'player': {'id': 5597, 'name': 'Keylor Navas Gamboa'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 5721, 'name': 'Daniel Carvajal Ramos'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 2}, {'player': {'id': 5485, 'name': 'Raphaël Varane'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 5}, {'player': {'id': 5201, 'name': 'Sergio Ramos García'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 5552, 'name': 'Marcelo Vieira da Silva Júnior'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 12}, {'player': {'id': 5539, 'name': 'Carlos Henrique Casimiro'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 5463, 'name': 'Luka Modrić'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 10}, {'player': {'id': 5574, 'name': 'Toni Kroos'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 8}, {'player': {'id': 4926, 'name': 'Francisco Román Alarcón Suárez'}, 'position': {'id': 19, 'name': 'Center Attacking Midfield'}, 'jersey_number': 22}, {'player': {'id': 19677, 'name': 'Karim Benzema'}, 'position': {'id': 22, 'name': 'Right Center Forward'}, 'jersey_number': 9}, {'player': {'id': 5207, 'name': 'Cristiano Ronaldo dos Santos Aveiro'}, 'position': {'id': 24, 'name': 'Left Center Forward'}, 'jersey_number': 7}]} |
```


```python
print(tact_Liv.to_markdown())
```

```
## |    | tactics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
## |---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
## |  1 | {'formation': 433, 'lineup': [{'player': {'id': 3630, 'name': 'Loris Karius'}, 'position': {'id': 1, 'name': 'Goalkeeper'}, 'jersey_number': 1}, {'player': {'id': 3664, 'name': 'Trent Alexander-Arnold'}, 'position': {'id': 2, 'name': 'Right Back'}, 'jersey_number': 66}, {'player': {'id': 3471, 'name': 'Dejan Lovren'}, 'position': {'id': 3, 'name': 'Right Center Back'}, 'jersey_number': 6}, {'player': {'id': 3669, 'name': 'Virgil van Dijk'}, 'position': {'id': 5, 'name': 'Left Center Back'}, 'jersey_number': 4}, {'player': {'id': 3655, 'name': 'Andrew Robertson'}, 'position': {'id': 6, 'name': 'Left Back'}, 'jersey_number': 26}, {'player': {'id': 3532, 'name': 'Jordan Brian Henderson'}, 'position': {'id': 10, 'name': 'Center Defensive Midfield'}, 'jersey_number': 14}, {'player': {'id': 3567, 'name': 'Georginio Wijnaldum'}, 'position': {'id': 13, 'name': 'Right Center Midfield'}, 'jersey_number': 5}, {'player': {'id': 3473, 'name': 'James Philip Milner'}, 'position': {'id': 15, 'name': 'Left Center Midfield'}, 'jersey_number': 7}, {'player': {'id': 3531, 'name': 'Mohamed Salah'}, 'position': {'id': 17, 'name': 'Right Wing'}, 'jersey_number': 11}, {'player': {'id': 3629, 'name': 'Sadio Mané'}, 'position': {'id': 21, 'name': 'Left Wing'}, 'jersey_number': 19}, {'player': {'id': 3535, 'name': 'Roberto Firmino Barbosa de Oliveira'}, 'position': {'id': 23, 'name': 'Center Forward'}, 'jersey_number': 9}]} |
```

So both `tact_Real` and `tact_Liv` are dataframes made of single rows with their indices (Which we will use to extract the data), and the `tactics` column is made up of a Python `dict` object. For now we are only interested in the key `'lineup'` to get all the information about the players from the teams. 


```python
dict_Real = tact_Real[0]['lineup']
dict_Liv = tact_Liv[1]['lineup']
```

We will use the `from_dict()` function provided by `pandas` to convert the dictionary into a dataframe.


```python
lineup_Real = pd.DataFrame.from_dict(dict_Real)
print(lineup_Real.to_markdown())
```

```
## |    | player                                                      | position                                        |   jersey_number |
## |---:|:------------------------------------------------------------|:------------------------------------------------|----------------:|
## |  0 | {'id': 5597, 'name': 'Keylor Navas Gamboa'}                 | {'id': 1, 'name': 'Goalkeeper'}                 |               1 |
## |  1 | {'id': 5721, 'name': 'Daniel Carvajal Ramos'}               | {'id': 2, 'name': 'Right Back'}                 |               2 |
## |  2 | {'id': 5485, 'name': 'Raphaël Varane'}                      | {'id': 3, 'name': 'Right Center Back'}          |               5 |
## |  3 | {'id': 5201, 'name': 'Sergio Ramos García'}                 | {'id': 5, 'name': 'Left Center Back'}           |               4 |
## |  4 | {'id': 5552, 'name': 'Marcelo Vieira da Silva Júnior'}      | {'id': 6, 'name': 'Left Back'}                  |              12 |
## |  5 | {'id': 5539, 'name': 'Carlos Henrique Casimiro'}            | {'id': 10, 'name': 'Center Defensive Midfield'} |              14 |
## |  6 | {'id': 5463, 'name': 'Luka Modrić'}                         | {'id': 13, 'name': 'Right Center Midfield'}     |              10 |
## |  7 | {'id': 5574, 'name': 'Toni Kroos'}                          | {'id': 15, 'name': 'Left Center Midfield'}      |               8 |
## |  8 | {'id': 4926, 'name': 'Francisco Román Alarcón Suárez'}      | {'id': 19, 'name': 'Center Attacking Midfield'} |              22 |
## |  9 | {'id': 19677, 'name': 'Karim Benzema'}                      | {'id': 22, 'name': 'Right Center Forward'}      |               9 |
## | 10 | {'id': 5207, 'name': 'Cristiano Ronaldo dos Santos Aveiro'} | {'id': 24, 'name': 'Left Center Forward'}       |               7 |
```


```python
lineup_Liv = pd.DataFrame.from_dict(dict_Liv)
print(lineup_Liv.to_markdown())
```

```
## |    | player                                                      | position                                        |   jersey_number |
## |---:|:------------------------------------------------------------|:------------------------------------------------|----------------:|
## |  0 | {'id': 3630, 'name': 'Loris Karius'}                        | {'id': 1, 'name': 'Goalkeeper'}                 |               1 |
## |  1 | {'id': 3664, 'name': 'Trent Alexander-Arnold'}              | {'id': 2, 'name': 'Right Back'}                 |              66 |
## |  2 | {'id': 3471, 'name': 'Dejan Lovren'}                        | {'id': 3, 'name': 'Right Center Back'}          |               6 |
## |  3 | {'id': 3669, 'name': 'Virgil van Dijk'}                     | {'id': 5, 'name': 'Left Center Back'}           |               4 |
## |  4 | {'id': 3655, 'name': 'Andrew Robertson'}                    | {'id': 6, 'name': 'Left Back'}                  |              26 |
## |  5 | {'id': 3532, 'name': 'Jordan Brian Henderson'}              | {'id': 10, 'name': 'Center Defensive Midfield'} |              14 |
## |  6 | {'id': 3567, 'name': 'Georginio Wijnaldum'}                 | {'id': 13, 'name': 'Right Center Midfield'}     |               5 |
## |  7 | {'id': 3473, 'name': 'James Philip Milner'}                 | {'id': 15, 'name': 'Left Center Midfield'}      |               7 |
## |  8 | {'id': 3531, 'name': 'Mohamed Salah'}                       | {'id': 17, 'name': 'Right Wing'}                |              11 |
## |  9 | {'id': 3629, 'name': 'Sadio Mané'}                          | {'id': 21, 'name': 'Left Wing'}                 |              19 |
## | 10 | {'id': 3535, 'name': 'Roberto Firmino Barbosa de Oliveira'} | {'id': 23, 'name': 'Center Forward'}            |               9 |
```

We are basically interested in the players name and their corresponding jersey numbers. We will use a simple for loop and store the information in seperate dictionaries for both the teams.


```python
players_Real = {}
for i in range(len(lineup_Real)):
    key = lineup_Real.player[i]['name']
    val = lineup_Real.jersey_number[i]
    players_Real[key] = val
print(players_Real)
```

```
## {'Keylor Navas Gamboa': 1, 'Daniel Carvajal Ramos': 2, 'Raphaël Varane': 5, 'Sergio Ramos García': 4, 'Marcelo Vieira da Silva Júnior': 12, 'Carlos Henrique Casimiro': 14, 'Luka Modrić': 10, 'Toni Kroos': 8, 'Francisco Román Alarcón Suárez': 22, 'Karim Benzema': 9, 'Cristiano Ronaldo dos Santos Aveiro': 7}
```


```python
players_Liv = {}
for i in range(len(lineup_Liv)):
    key = lineup_Liv.player[i]['name']
    val = lineup_Liv.jersey_number[i]
    players_Liv[key] = val
print(players_Liv)
```

```
## {'Loris Karius': 1, 'Trent Alexander-Arnold': 66, 'Dejan Lovren': 6, 'Virgil van Dijk': 4, 'Andrew Robertson': 26, 'Jordan Brian Henderson': 14, 'Georginio Wijnaldum': 5, 'James Philip Milner': 7, 'Mohamed Salah': 11, 'Sadio Mané': 19, 'Roberto Firmino Barbosa de Oliveira': 9}
```

So, we have collected the names and the jersey number of the players (starting 11) from both the teams in separate dictionaries named `players_Real` and `players_Liv`. These will come handy later!

Now from the `events` dataset we will extract out the relevant columns for our pass network analysis purposes.


```python
events_pn = events[['minute', 'second', 'team', 'type', 'location', 'pass_end_location', 'pass_outcome', 'player']]
```


```python
print(events_pn.head(10).to_markdown())
```

```
## |    |   minute |   second | team        | type        | location     | pass_end_location   | pass_outcome   | player              |
## |---:|---------:|---------:|:------------|:------------|:-------------|:--------------------|:---------------|:--------------------|
## |  0 |        0 |        0 | Real Madrid | Starting XI | nan          | nan                 | nan            | nan                 |
## |  1 |        0 |        0 | Liverpool   | Starting XI | nan          | nan                 | nan            | nan                 |
## |  2 |        0 |        0 | Real Madrid | Half Start  | nan          | nan                 | nan            | nan                 |
## |  3 |        0 |        0 | Liverpool   | Half Start  | nan          | nan                 | nan            | nan                 |
## |  4 |       45 |        0 | Liverpool   | Half Start  | nan          | nan                 | nan            | nan                 |
## |  5 |       45 |        0 | Real Madrid | Half Start  | nan          | nan                 | nan            | nan                 |
## |  6 |        0 |        0 | Liverpool   | Pass        | [60.0, 40.0] | [32.1, 41.2]        | nan            | James Philip Milner |
## |  7 |        0 |        3 | Liverpool   | Pass        | [35.0, 40.8] | [92.7, 22.7]        | Incomplete     | Dejan Lovren        |
## |  8 |        0 |        8 | Real Madrid | Pass        | [27.4, 60.2] | [36.1, 71.6]        | nan            | Raphaël Varane      |
## |  9 |        0 |       10 | Real Madrid | Pass        | [35.3, 75.4] | [22.4, 76.6]        | nan            | Luka Modrić         |
```


```python
print(events_pn.tail(10).to_markdown())
```

```
## |      |   minute |   second | team        | type           | location      |   pass_end_location |   pass_outcome | player              |
## |-----:|---------:|---------:|:------------|:---------------|:--------------|--------------------:|---------------:|:--------------------|
## | 3487 |       82 |       27 | Liverpool   | Substitution   | nan           |                 nan |            nan | James Philip Milner |
## | 3488 |       88 |       21 | Real Madrid | Substitution   | nan           |                 nan |            nan | Karim Benzema       |
## | 3489 |       31 |       41 | Liverpool   | Tactical Shift | nan           |                 nan |            nan | nan                 |
## | 3490 |       61 |        1 | Real Madrid | Tactical Shift | nan           |                 nan |            nan | nan                 |
## | 3491 |       88 |       34 | Real Madrid | Tactical Shift | nan           |                 nan |            nan | nan                 |
## | 3492 |       42 |       21 | Real Madrid | Offside        | [114.8, 41.4] |                 nan |            nan | Karim Benzema       |
## | 3493 |       48 |       31 | Real Madrid | Half End       | nan           |                 nan |            nan | nan                 |
## | 3494 |       48 |       31 | Liverpool   | Half End       | nan           |                 nan |            nan | nan                 |
## | 3495 |       93 |        2 | Liverpool   | Half End       | nan           |                 nan |            nan | nan                 |
## | 3496 |       93 |        2 | Real Madrid | Half End       | nan           |                 nan |            nan | nan                 |
```

The next step is to filter the datset by teams and store them as new datasets:


```python
events_Real = events_pn[events_pn['team'] == 'Real Madrid']
events_Liv = events_pn[events_pn['team'] == 'Liverpool']
```

View the first 10 rows from both the datasets:


```python
print(events_Real.head(10).to_markdown())
```

```
## |    |   minute |   second | team        | type        | location     | pass_end_location   | pass_outcome   | player                         |
## |---:|---------:|---------:|:------------|:------------|:-------------|:--------------------|:---------------|:-------------------------------|
## |  0 |        0 |        0 | Real Madrid | Starting XI | nan          | nan                 | nan            | nan                            |
## |  2 |        0 |        0 | Real Madrid | Half Start  | nan          | nan                 | nan            | nan                            |
## |  5 |       45 |        0 | Real Madrid | Half Start  | nan          | nan                 | nan            | nan                            |
## |  8 |        0 |        8 | Real Madrid | Pass        | [27.4, 60.2] | [36.1, 71.6]        | nan            | Raphaël Varane                 |
## |  9 |        0 |       10 | Real Madrid | Pass        | [35.3, 75.4] | [22.4, 76.6]        | nan            | Luka Modrić                    |
## | 10 |        0 |       11 | Real Madrid | Pass        | [22.3, 76.6] | [33.4, 68.0]        | nan            | Daniel Carvajal Ramos          |
## | 11 |        0 |       15 | Real Madrid | Pass        | [36.2, 75.3] | [43.6, 62.0]        | Incomplete     | Carlos Henrique Casimiro       |
## | 16 |        0 |       25 | Real Madrid | Pass        | [14.7, 23.2] | [56.7, 6.2]         | Incomplete     | Sergio Ramos García            |
## | 17 |        0 |       40 | Real Madrid | Pass        | [57.5, 4.6]  | [49.2, 15.6]        | nan            | Marcelo Vieira da Silva Júnior |
## | 18 |        0 |       43 | Real Madrid | Pass        | [48.8, 18.4] | [49.8, 12.5]        | nan            | Carlos Henrique Casimiro       |
```

```python
print(events_Liv.head(10).to_markdown())
```

```
## |    |   minute |   second | team      | type        | location     | pass_end_location   | pass_outcome   | player                              |
## |---:|---------:|---------:|:----------|:------------|:-------------|:--------------------|:---------------|:------------------------------------|
## |  1 |        0 |        0 | Liverpool | Starting XI | nan          | nan                 | nan            | nan                                 |
## |  3 |        0 |        0 | Liverpool | Half Start  | nan          | nan                 | nan            | nan                                 |
## |  4 |       45 |        0 | Liverpool | Half Start  | nan          | nan                 | nan            | nan                                 |
## |  6 |        0 |        0 | Liverpool | Pass        | [60.0, 40.0] | [32.1, 41.2]        | nan            | James Philip Milner                 |
## |  7 |        0 |        3 | Liverpool | Pass        | [35.0, 40.8] | [92.7, 22.7]        | Incomplete     | Dejan Lovren                        |
## | 12 |        0 |       16 | Liverpool | Pass        | [76.5, 18.1] | [84.8, 9.5]         | nan            | Jordan Brian Henderson              |
## | 13 |        0 |       18 | Liverpool | Pass        | [84.4, 10.0] | [92.5, 19.1]        | nan            | Sadio Mané                          |
## | 14 |        0 |       19 | Liverpool | Pass        | [91.6, 21.3] | [90.6, 50.7]        | nan            | Roberto Firmino Barbosa de Oliveira |
## | 15 |        0 |       22 | Liverpool | Pass        | [92.2, 50.9] | [109.7, 46.4]       | Incomplete     | Mohamed Salah                       |
## | 25 |        1 |        7 | Liverpool | Pass        | [42.0, 75.9] | [115.6, 59.3]       | Incomplete     | Trent Alexander-Arnold              |
```

As we are only interested in the pass network generation, we will filter the datasets by keeping those rows where `type` is set to `Pass`.


```python
events_pn_Real = events_Real[events_Real['type'] == 'Pass']
events_pn_Liv = events_Liv[events_Liv['type'] == 'Pass']
```

Again view the first 10 rows of the filtered datasets:


```python
print(events_pn_Real.head(10).to_markdown())
```

```
## |    |   minute |   second | team        | type   | location     | pass_end_location   | pass_outcome   | player                         |
## |---:|---------:|---------:|:------------|:-------|:-------------|:--------------------|:---------------|:-------------------------------|
## |  8 |        0 |        8 | Real Madrid | Pass   | [27.4, 60.2] | [36.1, 71.6]        | nan            | Raphaël Varane                 |
## |  9 |        0 |       10 | Real Madrid | Pass   | [35.3, 75.4] | [22.4, 76.6]        | nan            | Luka Modrić                    |
## | 10 |        0 |       11 | Real Madrid | Pass   | [22.3, 76.6] | [33.4, 68.0]        | nan            | Daniel Carvajal Ramos          |
## | 11 |        0 |       15 | Real Madrid | Pass   | [36.2, 75.3] | [43.6, 62.0]        | Incomplete     | Carlos Henrique Casimiro       |
## | 16 |        0 |       25 | Real Madrid | Pass   | [14.7, 23.2] | [56.7, 6.2]         | Incomplete     | Sergio Ramos García            |
## | 17 |        0 |       40 | Real Madrid | Pass   | [57.5, 4.6]  | [49.2, 15.6]        | nan            | Marcelo Vieira da Silva Júnior |
## | 18 |        0 |       43 | Real Madrid | Pass   | [48.8, 18.4] | [49.8, 12.5]        | nan            | Carlos Henrique Casimiro       |
## | 19 |        0 |       46 | Real Madrid | Pass   | [48.8, 13.9] | [36.1, 56.3]        | nan            | Toni Kroos                     |
## | 20 |        0 |       52 | Real Madrid | Pass   | [41.3, 54.8] | [34.4, 40.2]        | nan            | Raphaël Varane                 |
## | 21 |        0 |       55 | Real Madrid | Pass   | [39.1, 36.5] | [65.4, 13.1]        | nan            | Sergio Ramos García            |
```

```python
print(events_pn_Liv.head(10).to_markdown())
```

```
## |    |   minute |   second | team      | type   | location     | pass_end_location   | pass_outcome   | player                              |
## |---:|---------:|---------:|:----------|:-------|:-------------|:--------------------|:---------------|:------------------------------------|
## |  6 |        0 |        0 | Liverpool | Pass   | [60.0, 40.0] | [32.1, 41.2]        | nan            | James Philip Milner                 |
## |  7 |        0 |        3 | Liverpool | Pass   | [35.0, 40.8] | [92.7, 22.7]        | Incomplete     | Dejan Lovren                        |
## | 12 |        0 |       16 | Liverpool | Pass   | [76.5, 18.1] | [84.8, 9.5]         | nan            | Jordan Brian Henderson              |
## | 13 |        0 |       18 | Liverpool | Pass   | [84.4, 10.0] | [92.5, 19.1]        | nan            | Sadio Mané                          |
## | 14 |        0 |       19 | Liverpool | Pass   | [91.6, 21.3] | [90.6, 50.7]        | nan            | Roberto Firmino Barbosa de Oliveira |
## | 15 |        0 |       22 | Liverpool | Pass   | [92.2, 50.9] | [109.7, 46.4]       | Incomplete     | Mohamed Salah                       |
## | 25 |        1 |        7 | Liverpool | Pass   | [42.0, 75.9] | [115.6, 59.3]       | Incomplete     | Trent Alexander-Arnold              |
## | 37 |        2 |        0 | Liverpool | Pass   | [9.9, 39.1]  | [28.1, 4.2]         | nan            | Virgil van Dijk                     |
## | 38 |        2 |        3 | Liverpool | Pass   | [43.2, 2.8]  | [50.1, 4.8]         | Incomplete     | Andrew Robertson                    |
## | 39 |        2 |        7 | Liverpool | Pass   | [53.2, 0.1]  | [50.0, 4.0]         | nan            | Andrew Robertson                    |
```

Let us now very carefully observe the datasets. Suppose from the `events_rn_Real` dataset, we are focusing on the second and the third row (index `1` and `2`). `Luka Modrić` makes the pass at around `0`th `minute` and `10`th `second` (Second row) and `Daniel Carvajal Ramos ` receives the pass at around `0`th `minute` and `11`th `second` (third row). So in both the datasets we need to add two extra columns named as `pass_maker` and `pass_receiver`, where `pass_maker` column would be similar to `player` column and the `pass_receiver` column would be the `player` column whose index would be shifted by one place in the negative direction. This can be achieved by the `shift()` function provided by `pandas`. We will perform this operation on both `events_pn_Real` and `events_pn_Liv`.


```python
events_pn_Real['pass_maker'] = events_pn_Real['player']
```

```
## C:/Users/indra/AppData/Local/r-miniconda/envs/r-reticulate/python.exe:1: SettingWithCopyWarning: 
## A value is trying to be set on a copy of a slice from a DataFrame.
## Try using .loc[row_indexer,col_indexer] = value instead
## 
## See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
```

```python
events_pn_Real['pass_receiver'] = events_pn_Real['player'].shift(-1)

events_pn_Liv['pass_maker'] = events_pn_Liv['player']
events_pn_Liv['pass_receiver'] = events_pn_Liv['player'].shift(-1)
```

Let us check now how the modified datasets look:


```python
print(events_pn_Real.head(10).to_markdown())
```

```
## |    |   minute |   second | team        | type   | location     | pass_end_location   | pass_outcome   | player                         | pass_maker                     | pass_receiver                       |
## |---:|---------:|---------:|:------------|:-------|:-------------|:--------------------|:---------------|:-------------------------------|:-------------------------------|:------------------------------------|
## |  8 |        0 |        8 | Real Madrid | Pass   | [27.4, 60.2] | [36.1, 71.6]        | nan            | Raphaël Varane                 | Raphaël Varane                 | Luka Modrić                         |
## |  9 |        0 |       10 | Real Madrid | Pass   | [35.3, 75.4] | [22.4, 76.6]        | nan            | Luka Modrić                    | Luka Modrić                    | Daniel Carvajal Ramos               |
## | 10 |        0 |       11 | Real Madrid | Pass   | [22.3, 76.6] | [33.4, 68.0]        | nan            | Daniel Carvajal Ramos          | Daniel Carvajal Ramos          | Carlos Henrique Casimiro            |
## | 11 |        0 |       15 | Real Madrid | Pass   | [36.2, 75.3] | [43.6, 62.0]        | Incomplete     | Carlos Henrique Casimiro       | Carlos Henrique Casimiro       | Sergio Ramos García                 |
## | 16 |        0 |       25 | Real Madrid | Pass   | [14.7, 23.2] | [56.7, 6.2]         | Incomplete     | Sergio Ramos García            | Sergio Ramos García            | Marcelo Vieira da Silva Júnior      |
## | 17 |        0 |       40 | Real Madrid | Pass   | [57.5, 4.6]  | [49.2, 15.6]        | nan            | Marcelo Vieira da Silva Júnior | Marcelo Vieira da Silva Júnior | Carlos Henrique Casimiro            |
## | 18 |        0 |       43 | Real Madrid | Pass   | [48.8, 18.4] | [49.8, 12.5]        | nan            | Carlos Henrique Casimiro       | Carlos Henrique Casimiro       | Toni Kroos                          |
## | 19 |        0 |       46 | Real Madrid | Pass   | [48.8, 13.9] | [36.1, 56.3]        | nan            | Toni Kroos                     | Toni Kroos                     | Raphaël Varane                      |
## | 20 |        0 |       52 | Real Madrid | Pass   | [41.3, 54.8] | [34.4, 40.2]        | nan            | Raphaël Varane                 | Raphaël Varane                 | Sergio Ramos García                 |
## | 21 |        0 |       55 | Real Madrid | Pass   | [39.1, 36.5] | [65.4, 13.1]        | nan            | Sergio Ramos García            | Sergio Ramos García            | Cristiano Ronaldo dos Santos Aveiro |
```

```python
print(events_pn_Liv.head(10).to_markdown())
```

```
## |    |   minute |   second | team      | type   | location     | pass_end_location   | pass_outcome   | player                              | pass_maker                          | pass_receiver                       |
## |---:|---------:|---------:|:----------|:-------|:-------------|:--------------------|:---------------|:------------------------------------|:------------------------------------|:------------------------------------|
## |  6 |        0 |        0 | Liverpool | Pass   | [60.0, 40.0] | [32.1, 41.2]        | nan            | James Philip Milner                 | James Philip Milner                 | Dejan Lovren                        |
## |  7 |        0 |        3 | Liverpool | Pass   | [35.0, 40.8] | [92.7, 22.7]        | Incomplete     | Dejan Lovren                        | Dejan Lovren                        | Jordan Brian Henderson              |
## | 12 |        0 |       16 | Liverpool | Pass   | [76.5, 18.1] | [84.8, 9.5]         | nan            | Jordan Brian Henderson              | Jordan Brian Henderson              | Sadio Mané                          |
## | 13 |        0 |       18 | Liverpool | Pass   | [84.4, 10.0] | [92.5, 19.1]        | nan            | Sadio Mané                          | Sadio Mané                          | Roberto Firmino Barbosa de Oliveira |
## | 14 |        0 |       19 | Liverpool | Pass   | [91.6, 21.3] | [90.6, 50.7]        | nan            | Roberto Firmino Barbosa de Oliveira | Roberto Firmino Barbosa de Oliveira | Mohamed Salah                       |
## | 15 |        0 |       22 | Liverpool | Pass   | [92.2, 50.9] | [109.7, 46.4]       | Incomplete     | Mohamed Salah                       | Mohamed Salah                       | Trent Alexander-Arnold              |
## | 25 |        1 |        7 | Liverpool | Pass   | [42.0, 75.9] | [115.6, 59.3]       | Incomplete     | Trent Alexander-Arnold              | Trent Alexander-Arnold              | Virgil van Dijk                     |
## | 37 |        2 |        0 | Liverpool | Pass   | [9.9, 39.1]  | [28.1, 4.2]         | nan            | Virgil van Dijk                     | Virgil van Dijk                     | Andrew Robertson                    |
## | 38 |        2 |        3 | Liverpool | Pass   | [43.2, 2.8]  | [50.1, 4.8]         | Incomplete     | Andrew Robertson                    | Andrew Robertson                    | Andrew Robertson                    |
## | 39 |        2 |        7 | Liverpool | Pass   | [53.2, 0.1]  | [50.0, 4.0]         | nan            | Andrew Robertson                    | Andrew Robertson                    | James Philip Milner                 |
```

Now, there might be passes which were not successful. Remember from the [third post](https://realsoccerexpand.netlify.app/post/pass-map-shot-map-and-heat-map/) that in the statsbomb data passes whose `pass_outcome` are set as `nan` are actually the successful passes. We will again filter the datasets by successful passes:


```python
events_pn_Real = events_pn_Real[events_pn_Real['pass_outcome'].isnull() == True].reset_index()
events_pn_Liv = events_pn_Liv[events_pn_Liv['pass_outcome'].isnull() == True].reset_index()
```

The first 10 rows of the filtered datasets:


```python
print(events_pn_Real.head(10).to_markdown())
```

```
## |    |   index |   minute |   second | team        | type   | location     | pass_end_location   |   pass_outcome | player                              | pass_maker                          | pass_receiver                       |
## |---:|--------:|---------:|---------:|:------------|:-------|:-------------|:--------------------|---------------:|:------------------------------------|:------------------------------------|:------------------------------------|
## |  0 |       8 |        0 |        8 | Real Madrid | Pass   | [27.4, 60.2] | [36.1, 71.6]        |            nan | Raphaël Varane                      | Raphaël Varane                      | Luka Modrić                         |
## |  1 |       9 |        0 |       10 | Real Madrid | Pass   | [35.3, 75.4] | [22.4, 76.6]        |            nan | Luka Modrić                         | Luka Modrić                         | Daniel Carvajal Ramos               |
## |  2 |      10 |        0 |       11 | Real Madrid | Pass   | [22.3, 76.6] | [33.4, 68.0]        |            nan | Daniel Carvajal Ramos               | Daniel Carvajal Ramos               | Carlos Henrique Casimiro            |
## |  3 |      17 |        0 |       40 | Real Madrid | Pass   | [57.5, 4.6]  | [49.2, 15.6]        |            nan | Marcelo Vieira da Silva Júnior      | Marcelo Vieira da Silva Júnior      | Carlos Henrique Casimiro            |
## |  4 |      18 |        0 |       43 | Real Madrid | Pass   | [48.8, 18.4] | [49.8, 12.5]        |            nan | Carlos Henrique Casimiro            | Carlos Henrique Casimiro            | Toni Kroos                          |
## |  5 |      19 |        0 |       46 | Real Madrid | Pass   | [48.8, 13.9] | [36.1, 56.3]        |            nan | Toni Kroos                          | Toni Kroos                          | Raphaël Varane                      |
## |  6 |      20 |        0 |       52 | Real Madrid | Pass   | [41.3, 54.8] | [34.4, 40.2]        |            nan | Raphaël Varane                      | Raphaël Varane                      | Sergio Ramos García                 |
## |  7 |      21 |        0 |       55 | Real Madrid | Pass   | [39.1, 36.5] | [65.4, 13.1]        |            nan | Sergio Ramos García                 | Sergio Ramos García                 | Cristiano Ronaldo dos Santos Aveiro |
## |  8 |      22 |        0 |       58 | Real Madrid | Pass   | [64.5, 11.1] | [54.2, 5.6]         |            nan | Cristiano Ronaldo dos Santos Aveiro | Cristiano Ronaldo dos Santos Aveiro | Marcelo Vieira da Silva Júnior      |
## |  9 |      23 |        0 |       59 | Real Madrid | Pass   | [55.3, 5.5]  | [83.9, 4.3]         |            nan | Marcelo Vieira da Silva Júnior      | Marcelo Vieira da Silva Júnior      | Karim Benzema                       |
```

```python
print(events_pn_Liv.head(10).to_markdown())
```

```
## |    |   index |   minute |   second | team      | type   | location     | pass_end_location   |   pass_outcome | player                              | pass_maker                          | pass_receiver                       |
## |---:|--------:|---------:|---------:|:----------|:-------|:-------------|:--------------------|---------------:|:------------------------------------|:------------------------------------|:------------------------------------|
## |  0 |       6 |        0 |        0 | Liverpool | Pass   | [60.0, 40.0] | [32.1, 41.2]        |            nan | James Philip Milner                 | James Philip Milner                 | Dejan Lovren                        |
## |  1 |      12 |        0 |       16 | Liverpool | Pass   | [76.5, 18.1] | [84.8, 9.5]         |            nan | Jordan Brian Henderson              | Jordan Brian Henderson              | Sadio Mané                          |
## |  2 |      13 |        0 |       18 | Liverpool | Pass   | [84.4, 10.0] | [92.5, 19.1]        |            nan | Sadio Mané                          | Sadio Mané                          | Roberto Firmino Barbosa de Oliveira |
## |  3 |      14 |        0 |       19 | Liverpool | Pass   | [91.6, 21.3] | [90.6, 50.7]        |            nan | Roberto Firmino Barbosa de Oliveira | Roberto Firmino Barbosa de Oliveira | Mohamed Salah                       |
## |  4 |      37 |        2 |        0 | Liverpool | Pass   | [9.9, 39.1]  | [28.1, 4.2]         |            nan | Virgil van Dijk                     | Virgil van Dijk                     | Andrew Robertson                    |
## |  5 |      39 |        2 |        7 | Liverpool | Pass   | [53.2, 0.1]  | [50.0, 4.0]         |            nan | Andrew Robertson                    | Andrew Robertson                    | James Philip Milner                 |
## |  6 |      40 |        2 |       10 | Liverpool | Pass   | [45.5, 4.0]  | [27.4, 16.8]        |            nan | James Philip Milner                 | James Philip Milner                 | Virgil van Dijk                     |
## |  7 |      41 |        2 |       13 | Liverpool | Pass   | [26.7, 19.6] | [27.8, 47.3]        |            nan | Virgil van Dijk                     | Virgil van Dijk                     | Dejan Lovren                        |
## |  8 |      42 |        2 |       16 | Liverpool | Pass   | [28.0, 45.4] | [28.4, 21.4]        |            nan | Dejan Lovren                        | Dejan Lovren                        | Virgil van Dijk                     |
## |  9 |      43 |        2 |       19 | Liverpool | Pass   | [30.4, 25.7] | [30.7, 52.9]        |            nan | Virgil van Dijk                     | Virgil van Dijk                     | Dejan Lovren                        |
```

So it seems we have been able to logically clean and modify the datasets. Now we are only focused on building the pass netwrok among the players who were in the starting 11 from both the teams. So we will discard out the rows which consist of pass events that took place after the first substitution for either of the teams. Let us find the `minute` and `second` of the first substitution for both `Real Madrid` and `Barcelona`.

So let us filter the datasets `events_Real` and `events_Liv` by setting the `type` to be `Substitution`. This will give us the information of when the first substitution had taken place for the teams.


```python
substitution_Real = events_Real[events_Real['type'] == 'Substitution']
substitution_Liv = events_Liv[events_Liv['type'] == 'Substitution']
```

And let us view the datasets:


```python
print(substitution_Real.to_markdown())
```

```
## |      |   minute |   second | team        | type         |   location |   pass_end_location |   pass_outcome | player                         |
## |-----:|---------:|---------:|:------------|:-------------|-----------:|--------------------:|---------------:|:-------------------------------|
## | 3485 |       36 |       17 | Real Madrid | Substitution |        nan |                 nan |            nan | Daniel Carvajal Ramos          |
## | 3486 |       60 |       56 | Real Madrid | Substitution |        nan |                 nan |            nan | Francisco Román Alarcón Suárez |
## | 3488 |       88 |       21 | Real Madrid | Substitution |        nan |                 nan |            nan | Karim Benzema                  |
```

```python
print(substitution_Liv.to_markdown())
```

```
## |      |   minute |   second | team      | type         |   location |   pass_end_location |   pass_outcome | player              |
## |-----:|---------:|---------:|:----------|:-------------|-----------:|--------------------:|---------------:|:--------------------|
## | 3484 |       29 |       39 | Liverpool | Substitution |        nan |                 nan |            nan | Mohamed Salah       |
## | 3487 |       82 |       27 | Liverpool | Substitution |        nan |                 nan |            nan | James Philip Milner |
```

We see that the first substitution takes place for `Real Madrid` at the `36`th minute and `17`th second, whereas for `Liverpool` it takes place around `29`th minute and `39`th second. Let us find these out by writing a small Python code:


```python
substitution_Real_minute = np.min(substitution_Real['minute'])
substitution_Real_minute_data = substitution_Real[substitution_Real['minute'] == substitution_Real_minute]
substitution_Real_second = np.min(substitution_Real_minute_data['second'])
print("minute =", substitution_Real_minute, "second =",  substitution_Real_second)
```

```
## minute = 36 second = 17
```


```python
substitution_Liv_minute = np.min(substitution_Liv['minute'])
substitution_Liv_minute_data = substitution_Liv[substitution_Liv['minute'] == substitution_Liv_minute]
substitution_Liv_second = np.min(substitution_Liv_minute_data['second'])
print("minute = ", substitution_Liv_minute, "second = ", substitution_Liv_second)
```

```
## minute =  29 second =  39
```

We see that we have gotten the correct timings of when the first substitutions had taken place. Now we filter our datasets by taking tose pass events that took place before the first substitutions 


```python
events_pn_Real = events_pn_Real[(events_pn_Real['minute'] <= substitution_Real_minute)]

events_pn_Liv = events_pn_Liv[(events_pn_Liv['minute'] <= substitution_Liv_minute)]
```

Let us again print the first 10 rows of the renewed datasets:


```python
print(events_pn_Real.head(10).to_markdown())
```

```
## |    |   index |   minute |   second | team        | type   | location     | pass_end_location   |   pass_outcome | player                              | pass_maker                          | pass_receiver                       |
## |---:|--------:|---------:|---------:|:------------|:-------|:-------------|:--------------------|---------------:|:------------------------------------|:------------------------------------|:------------------------------------|
## |  0 |       8 |        0 |        8 | Real Madrid | Pass   | [27.4, 60.2] | [36.1, 71.6]        |            nan | Raphaël Varane                      | Raphaël Varane                      | Luka Modrić                         |
## |  1 |       9 |        0 |       10 | Real Madrid | Pass   | [35.3, 75.4] | [22.4, 76.6]        |            nan | Luka Modrić                         | Luka Modrić                         | Daniel Carvajal Ramos               |
## |  2 |      10 |        0 |       11 | Real Madrid | Pass   | [22.3, 76.6] | [33.4, 68.0]        |            nan | Daniel Carvajal Ramos               | Daniel Carvajal Ramos               | Carlos Henrique Casimiro            |
## |  3 |      17 |        0 |       40 | Real Madrid | Pass   | [57.5, 4.6]  | [49.2, 15.6]        |            nan | Marcelo Vieira da Silva Júnior      | Marcelo Vieira da Silva Júnior      | Carlos Henrique Casimiro            |
## |  4 |      18 |        0 |       43 | Real Madrid | Pass   | [48.8, 18.4] | [49.8, 12.5]        |            nan | Carlos Henrique Casimiro            | Carlos Henrique Casimiro            | Toni Kroos                          |
## |  5 |      19 |        0 |       46 | Real Madrid | Pass   | [48.8, 13.9] | [36.1, 56.3]        |            nan | Toni Kroos                          | Toni Kroos                          | Raphaël Varane                      |
## |  6 |      20 |        0 |       52 | Real Madrid | Pass   | [41.3, 54.8] | [34.4, 40.2]        |            nan | Raphaël Varane                      | Raphaël Varane                      | Sergio Ramos García                 |
## |  7 |      21 |        0 |       55 | Real Madrid | Pass   | [39.1, 36.5] | [65.4, 13.1]        |            nan | Sergio Ramos García                 | Sergio Ramos García                 | Cristiano Ronaldo dos Santos Aveiro |
## |  8 |      22 |        0 |       58 | Real Madrid | Pass   | [64.5, 11.1] | [54.2, 5.6]         |            nan | Cristiano Ronaldo dos Santos Aveiro | Cristiano Ronaldo dos Santos Aveiro | Marcelo Vieira da Silva Júnior      |
## |  9 |      23 |        0 |       59 | Real Madrid | Pass   | [55.3, 5.5]  | [83.9, 4.3]         |            nan | Marcelo Vieira da Silva Júnior      | Marcelo Vieira da Silva Júnior      | Karim Benzema                       |
```

```python
print(events_pn_Liv.head(10).to_markdown())
```

```
## |    |   index |   minute |   second | team      | type   | location     | pass_end_location   |   pass_outcome | player                              | pass_maker                          | pass_receiver                       |
## |---:|--------:|---------:|---------:|:----------|:-------|:-------------|:--------------------|---------------:|:------------------------------------|:------------------------------------|:------------------------------------|
## |  0 |       6 |        0 |        0 | Liverpool | Pass   | [60.0, 40.0] | [32.1, 41.2]        |            nan | James Philip Milner                 | James Philip Milner                 | Dejan Lovren                        |
## |  1 |      12 |        0 |       16 | Liverpool | Pass   | [76.5, 18.1] | [84.8, 9.5]         |            nan | Jordan Brian Henderson              | Jordan Brian Henderson              | Sadio Mané                          |
## |  2 |      13 |        0 |       18 | Liverpool | Pass   | [84.4, 10.0] | [92.5, 19.1]        |            nan | Sadio Mané                          | Sadio Mané                          | Roberto Firmino Barbosa de Oliveira |
## |  3 |      14 |        0 |       19 | Liverpool | Pass   | [91.6, 21.3] | [90.6, 50.7]        |            nan | Roberto Firmino Barbosa de Oliveira | Roberto Firmino Barbosa de Oliveira | Mohamed Salah                       |
## |  4 |      37 |        2 |        0 | Liverpool | Pass   | [9.9, 39.1]  | [28.1, 4.2]         |            nan | Virgil van Dijk                     | Virgil van Dijk                     | Andrew Robertson                    |
## |  5 |      39 |        2 |        7 | Liverpool | Pass   | [53.2, 0.1]  | [50.0, 4.0]         |            nan | Andrew Robertson                    | Andrew Robertson                    | James Philip Milner                 |
## |  6 |      40 |        2 |       10 | Liverpool | Pass   | [45.5, 4.0]  | [27.4, 16.8]        |            nan | James Philip Milner                 | James Philip Milner                 | Virgil van Dijk                     |
## |  7 |      41 |        2 |       13 | Liverpool | Pass   | [26.7, 19.6] | [27.8, 47.3]        |            nan | Virgil van Dijk                     | Virgil van Dijk                     | Dejan Lovren                        |
## |  8 |      42 |        2 |       16 | Liverpool | Pass   | [28.0, 45.4] | [28.4, 21.4]        |            nan | Dejan Lovren                        | Dejan Lovren                        | Virgil van Dijk                     |
## |  9 |      43 |        2 |       19 | Liverpool | Pass   | [30.4, 25.7] | [30.7, 52.9]        |            nan | Virgil van Dijk                     | Virgil van Dijk                     | Dejan Lovren                        |
```

Now from the datasets, we will split the `location` and the `pass_end_location` columns into two columns each representing the coordinates and name them as `pass_maker_x`, `pass_maker_y`, `pass_receiver_x` and `pass_receiver_y`.

Let us manipulate the dataset for `Real Madrid` first:


```python
Loc = events_pn_Real['location']
Loc = pd.DataFrame(Loc.to_list(), columns=['pass_maker_x', 'pass_maker_y'])

Loc_end = events_pn_Real['pass_end_location']
Loc_end = pd.DataFrame(Loc_end.to_list(), columns=['pass_receiver_x', 'pass_receiver_y'])

events_pn_Real['pass_maker_x'] = Loc['pass_maker_x']
events_pn_Real['pass_maker_y'] = Loc['pass_maker_y']
events_pn_Real['pass_receiver_x'] = Loc_end['pass_receiver_x']
events_pn_Real['pass_receiver_y'] = Loc_end['pass_receiver_y']

events_pn_Real = events_pn_Real[['minute', 'second', 'team', 'type', 'pass_outcome', 'player', 'pass_maker', 'pass_receiver', 'pass_maker_x', 'pass_maker_y', 'pass_receiver_x', 'pass_receiver_y']]

print(events_pn_Real.head(10).to_markdown())
```

```
## |    |   minute |   second | team        | type   |   pass_outcome | player                              | pass_maker                          | pass_receiver                       |   pass_maker_x |   pass_maker_y |   pass_receiver_x |   pass_receiver_y |
## |---:|---------:|---------:|:------------|:-------|---------------:|:------------------------------------|:------------------------------------|:------------------------------------|---------------:|---------------:|------------------:|------------------:|
## |  0 |        0 |        8 | Real Madrid | Pass   |            nan | Raphaël Varane                      | Raphaël Varane                      | Luka Modrić                         |           27.4 |           60.2 |              36.1 |              71.6 |
## |  1 |        0 |       10 | Real Madrid | Pass   |            nan | Luka Modrić                         | Luka Modrić                         | Daniel Carvajal Ramos               |           35.3 |           75.4 |              22.4 |              76.6 |
## |  2 |        0 |       11 | Real Madrid | Pass   |            nan | Daniel Carvajal Ramos               | Daniel Carvajal Ramos               | Carlos Henrique Casimiro            |           22.3 |           76.6 |              33.4 |              68   |
## |  3 |        0 |       40 | Real Madrid | Pass   |            nan | Marcelo Vieira da Silva Júnior      | Marcelo Vieira da Silva Júnior      | Carlos Henrique Casimiro            |           57.5 |            4.6 |              49.2 |              15.6 |
## |  4 |        0 |       43 | Real Madrid | Pass   |            nan | Carlos Henrique Casimiro            | Carlos Henrique Casimiro            | Toni Kroos                          |           48.8 |           18.4 |              49.8 |              12.5 |
## |  5 |        0 |       46 | Real Madrid | Pass   |            nan | Toni Kroos                          | Toni Kroos                          | Raphaël Varane                      |           48.8 |           13.9 |              36.1 |              56.3 |
## |  6 |        0 |       52 | Real Madrid | Pass   |            nan | Raphaël Varane                      | Raphaël Varane                      | Sergio Ramos García                 |           41.3 |           54.8 |              34.4 |              40.2 |
## |  7 |        0 |       55 | Real Madrid | Pass   |            nan | Sergio Ramos García                 | Sergio Ramos García                 | Cristiano Ronaldo dos Santos Aveiro |           39.1 |           36.5 |              65.4 |              13.1 |
## |  8 |        0 |       58 | Real Madrid | Pass   |            nan | Cristiano Ronaldo dos Santos Aveiro | Cristiano Ronaldo dos Santos Aveiro | Marcelo Vieira da Silva Júnior      |           64.5 |           11.1 |              54.2 |               5.6 |
## |  9 |        0 |       59 | Real Madrid | Pass   |            nan | Marcelo Vieira da Silva Júnior      | Marcelo Vieira da Silva Júnior      | Karim Benzema                       |           55.3 |            5.5 |              83.9 |               4.3 |
```

Same manipulation for Liverpool:


```python
Loc = events_pn_Liv['location']
Loc = pd.DataFrame(Loc.to_list(), columns=['pass_maker_x', 'pass_maker_y'])

Loc_end = events_pn_Liv['pass_end_location']
Loc_end = pd.DataFrame(Loc_end.to_list(), columns=['pass_receiver_x', 'pass_receiver_y'])

events_pn_Liv['pass_maker_x'] = Loc['pass_maker_x']
events_pn_Liv['pass_maker_y'] = Loc['pass_maker_y']
events_pn_Liv['pass_receiver_x'] = Loc_end['pass_receiver_x']
events_pn_Liv['pass_receiver_y'] = Loc_end['pass_receiver_y']

events_pn_Liv = events_pn_Liv[['minute', 'second', 'team', 'type', 'pass_outcome', 'player', 'pass_maker', 'pass_receiver', 'pass_maker_x', 'pass_maker_y', 'pass_receiver_x', 'pass_receiver_y']]

print(events_pn_Liv.head(10).to_markdown())
```

```
## |    |   minute |   second | team      | type   |   pass_outcome | player                              | pass_maker                          | pass_receiver                       |   pass_maker_x |   pass_maker_y |   pass_receiver_x |   pass_receiver_y |
## |---:|---------:|---------:|:----------|:-------|---------------:|:------------------------------------|:------------------------------------|:------------------------------------|---------------:|---------------:|------------------:|------------------:|
## |  0 |        0 |        0 | Liverpool | Pass   |            nan | James Philip Milner                 | James Philip Milner                 | Dejan Lovren                        |           60   |           40   |              32.1 |              41.2 |
## |  1 |        0 |       16 | Liverpool | Pass   |            nan | Jordan Brian Henderson              | Jordan Brian Henderson              | Sadio Mané                          |           76.5 |           18.1 |              84.8 |               9.5 |
## |  2 |        0 |       18 | Liverpool | Pass   |            nan | Sadio Mané                          | Sadio Mané                          | Roberto Firmino Barbosa de Oliveira |           84.4 |           10   |              92.5 |              19.1 |
## |  3 |        0 |       19 | Liverpool | Pass   |            nan | Roberto Firmino Barbosa de Oliveira | Roberto Firmino Barbosa de Oliveira | Mohamed Salah                       |           91.6 |           21.3 |              90.6 |              50.7 |
## |  4 |        2 |        0 | Liverpool | Pass   |            nan | Virgil van Dijk                     | Virgil van Dijk                     | Andrew Robertson                    |            9.9 |           39.1 |              28.1 |               4.2 |
## |  5 |        2 |        7 | Liverpool | Pass   |            nan | Andrew Robertson                    | Andrew Robertson                    | James Philip Milner                 |           53.2 |            0.1 |              50   |               4   |
## |  6 |        2 |       10 | Liverpool | Pass   |            nan | James Philip Milner                 | James Philip Milner                 | Virgil van Dijk                     |           45.5 |            4   |              27.4 |              16.8 |
## |  7 |        2 |       13 | Liverpool | Pass   |            nan | Virgil van Dijk                     | Virgil van Dijk                     | Dejan Lovren                        |           26.7 |           19.6 |              27.8 |              47.3 |
## |  8 |        2 |       16 | Liverpool | Pass   |            nan | Dejan Lovren                        | Dejan Lovren                        | Virgil van Dijk                     |           28   |           45.4 |              28.4 |              21.4 |
## |  9 |        2 |       19 | Liverpool | Pass   |            nan | Virgil van Dijk                     | Virgil van Dijk                     | Dejan Lovren                        |           30.4 |           25.7 |              30.7 |              52.9 |
```

Inspired by the way given [here](https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_plots/plot_pass_network.html), we will take the average locations of the starting 11 players on the field for a unified construction of the pass network, and also will count the number of passes created by these player:


```python
av_loc_Real = events_pn_Real.groupby('pass_maker').agg({'pass_maker_x':['mean'], 'pass_maker_y':['mean', 'count']})
print(av_loc_Real.to_markdown())
```

```
## | pass_maker                          |   ('pass_maker_x', 'mean') |   ('pass_maker_y', 'mean') |   ('pass_maker_y', 'count') |
## |:------------------------------------|---------------------------:|---------------------------:|----------------------------:|
## | Carlos Henrique Casimiro            |                    60.8455 |                    31.8364 |                          11 |
## | Cristiano Ronaldo dos Santos Aveiro |                    81.58   |                    29.16   |                          10 |
## | Daniel Carvajal Ramos               |                    64.3417 |                    73.875  |                          24 |
## | Francisco Román Alarcón Suárez      |                    62.3235 |                    27.0824 |                          17 |
## | Karim Benzema                       |                    65.0818 |                    27.9364 |                          11 |
## | Keylor Navas Gamboa                 |                    10.87   |                    41.81   |                          10 |
## | Luka Modrić                         |                    60.6048 |                    55.0286 |                          21 |
## | Marcelo Vieira da Silva Júnior      |                    59.8652 |                    11.1304 |                          23 |
## | Raphaël Varane                      |                    37.4364 |                    58.3545 |                          22 |
## | Sergio Ramos García                 |                    41.2824 |                    24.5147 |                          34 |
## | Toni Kroos                          |                    51.19   |                    24.275  |                          40 |
```

As we see the `groupby()` function from `pandas` splits `events_pn_Real` into groups indexed by the player names. Whereas, the `agg()` function aggregates the data into the averages of the pass makers' locations and also counts the number of passes made by these players. Now refine the column names of `av_loc_Real`:


```python
av_loc_Real.columns = ['passer_x', 'passer_y', 'count']
print(av_loc_Real.to_markdown())
```

```
## | pass_maker                          |   passer_x |   passer_y |   count |
## |:------------------------------------|-----------:|-----------:|--------:|
## | Carlos Henrique Casimiro            |    60.8455 |    31.8364 |      11 |
## | Cristiano Ronaldo dos Santos Aveiro |    81.58   |    29.16   |      10 |
## | Daniel Carvajal Ramos               |    64.3417 |    73.875  |      24 |
## | Francisco Román Alarcón Suárez      |    62.3235 |    27.0824 |      17 |
## | Karim Benzema                       |    65.0818 |    27.9364 |      11 |
## | Keylor Navas Gamboa                 |    10.87   |    41.81   |      10 |
## | Luka Modrić                         |    60.6048 |    55.0286 |      21 |
## | Marcelo Vieira da Silva Júnior      |    59.8652 |    11.1304 |      23 |
## | Raphaël Varane                      |    37.4364 |    58.3545 |      22 |
## | Sergio Ramos García                 |    41.2824 |    24.5147 |      34 |
## | Toni Kroos                          |    51.19   |    24.275  |      40 |
```

Now do the same operations for `Liverpool`:


```python
av_loc_Liv = events_pn_Liv.groupby('pass_maker').agg({'pass_maker_x':['mean'], 'pass_maker_y':['mean', 'count']})
av_loc_Liv.columns = ['passer_x', 'passer_y', 'count']
print(av_loc_Liv.to_markdown())
```

```
## | pass_maker                          |   passer_x |   passer_y |   count |
## |:------------------------------------|-----------:|-----------:|--------:|
## | Andrew Robertson                    |    59.8154 |    6.83077 |      13 |
## | Dejan Lovren                        |    41.6909 |   60.1727  |      11 |
## | Georginio Wijnaldum                 |    76.3909 |   28.5182  |      11 |
## | James Philip Milner                 |    72.3533 |   36.1533  |      15 |
## | Jordan Brian Henderson              |    61.0353 |   37.1529  |      17 |
## | Loris Karius                        |    12.9143 |   40.3857  |       7 |
## | Mohamed Salah                       |    77.55   |   64.71    |      10 |
## | Roberto Firmino Barbosa de Oliveira |    78.25   |   43.57    |      10 |
## | Sadio Mané                          |    86.275  |   22.075   |       4 |
## | Trent Alexander-Arnold              |    64.6667 |   72.55    |      12 |
## | Virgil van Dijk                     |    43.3667 |   25.4333  |       9 |
```

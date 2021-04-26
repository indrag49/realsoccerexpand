---
title: Draw a football pitch
author: Indranil Ghosh
date: '2021-04-26'
slug: visualize-a-pitch
categories: ["Python", "visualization"]
tags: ["football pitch", "mplsoccer"]
subtitle: 'using mplsoccer'
summary: ''
authors: []
lastmod: '2021-04-26T22:13:46+05:30'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

If you do not want to recreate a  football pitch manually using Python (which would be rather tedious) you can simply use the  [**mplsoccer**](https://mplsoccer.readthedocs.io/en/latest/index.html) module without any concern. To my knowledge it provides with the best functionalities to draw a football pitch. This package is maintained by  [Anmol Durgapal](https://twitter.com/slothfulwave612) and [Andrew Rowlinson](https://twitter.com/numberstorm).


Keep in mind you can do a lot more advanced visualization stuffs using **mplsoccer** besides drawing a football pitch. We will encounter them as we move forward with other posts later. For now let us focus on visualizing a pitch in the simplest way possible. We need to `pip` install the package first


```python
pip install mplsoccer
```

Note that `mplsoccer` uses Python 3.6+. Next we need to import `matplotlib` and the `Pitch` classes. 


```python
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
```

Let us try to draw the simplest football pitch that satisfies our visualization needs.


```python
pitch = Pitch(pitch_color='grass', line_color='white', stripe=True, constrained_layout=True,
        tight_layout= False, goal_type = 'box', label = True,  axis = True, tick=True)
fig, ax = pitch.draw()
plt.show()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-1.png" width="672" />


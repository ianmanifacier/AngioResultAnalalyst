# import progressbar
# from time import sleep
# bar = progressbar.ProgressBar(maxval=20, \
#     widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
# bar.start()
# for i in range(20):
#     bar.update(i+1)
#     sleep(0.1)
# bar.finish()

# source: https://towardsdatascience.com/least-squares-linear-regression-in-python-54b87fc49e77
# py -i draftplot.py (to keep script open)

import plotly.graph_objects as go
import numpy as np
np.random.seed(44)

y0 = np.random.randn(50) - 1
y1 = np.random.randn(50) + 1

fig = go.Figure()
fig.add_trace(go.Box(y=y0))
fig.add_trace(go.Box(y=y1))

fig.show()

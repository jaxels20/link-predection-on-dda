import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('gan.csv')

fig = go.Figure()

fig.add_trace(go.Scatter(x=df['epoch'], y=df['d_fake_loss'], name='d_fake_loss'))
fig.add_trace(go.Scatter(x=df['epoch'], y=df['d_real_loss'], name='d_real_loss'))
fig.add_trace(go.Scatter(x=df['epoch'], y=df['d_loss'], name='d_loss'))
fig.add_trace(go.Scatter(x=df['epoch'], y=df['g_loss'], name='g_loss'))

fig.show()

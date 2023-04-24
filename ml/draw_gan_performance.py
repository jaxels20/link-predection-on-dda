import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def draw_degree_dist(nested_arr, epoch, names):
    fig = go.Figure()
    # boxplot 
    for i, arr in enumerate(nested_arr):
        fig.add_trace(go.Box(y=arr, name=f'Box Plot {epoch, names[i]}', boxpoints='all', jitter=0.5))

    fig.show()





def draw_gan_performance():
    g_df = pd.read_csv('g.csv')
    d_df = pd.read_csv("d.csv")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g_df['overall_epoch'], y=g_df['loss'], name='Generator Loss'))
    fig.add_trace(go.Scatter(x=d_df['overall_epoch'], y=d_df['loss'], name='Discriminator Loss'))
    fig.add_trace(go.Scatter(x=d_df['overall_epoch'], y=d_df['acc'], name='Discriminator Accuracy'))
    fig.add_trace(go.Scatter(x=d_df['overall_epoch'], y=g_df['acc'], name='Generator Accuracy'))

    fig.show()

if __name__ == '__main__':
    draw_gan_performance()



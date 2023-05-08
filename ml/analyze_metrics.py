import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


metrics_postbatch = pd.read_csv('optim_metric_variance_postbatch.csv')
metrics_postbatch_second = pd.read_csv('optim_metric_variance_postbatch_second.csv')
metrics_prebatch = pd.read_csv('optim_metric_variance_prebatch.csv')

print('Mean of postbatch:', metrics_postbatch['auc'].mean())
print('Mean of postbatch second:', metrics_postbatch_second['auc'].mean())
print('Mean of prebatch:', metrics_prebatch['auc'].mean())

def plot_auc_variance():
    # Plot the variance of the AUC from both dataframes in one plot
    fig = go.Figure()
    fig.add_trace(go.Box(y=metrics_postbatch['auc'], name='Postbatch'))
    fig.add_trace(go.Box(y=metrics_prebatch['auc'], name='Prebatch'))
    fig.update_layout(
        xaxis_title='Batch',
        yaxis_title='AUC',
        font_family = "Moderne Computer"
    )
    fig.write_image("auc_variance.pdf")

#plot_auc_variance()
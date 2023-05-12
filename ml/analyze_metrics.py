import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


#metrics_postbatch = pd.read_csv('optim_metric_variance_postbatch.csv')
metrics_postbatch_maxaggr = pd.read_csv('optim_metric_variance_postbatch_maxaggr.csv')
metrics_prebatch_meanaggr = pd.read_csv('optim_metric_variance_prebatch_meanaggr.csv')
metrics_prebatch_gan_meanaggr = pd.read_csv('gan_model_results.csv')
#metrics_prebatch_maxaggr = pd.read_csv('optim_metric_variance_prebatch_maxaggr.csv')

#print('Mean of postbatch:', metrics_postbatch_maxaggr['auc'].mean())
#print('Mean of prebatch:', metrics_prebatch_maxaggr['auc'].mean())

def plot_auc_variance():
    # Plot the variance of the AUC from both dataframes in one plot
    fig = go.Figure()
    fig.add_trace(go.Box(y=metrics_prebatch_meanaggr['auc'], name='Prebatch'))
    fig.add_trace(go.Box(y=metrics_postbatch_maxaggr['auc'], name='Postbatch'))
    fig.update_layout(
        xaxis_title='Batch',
        yaxis_title='AUC',
        font_family = "Moderne Computer"
    )
    fig.write_image("auc_variation_mixedaggr.pdf")


def plot_all_metrics():
    # Plot all metrics for prebatch and postbatch in a box plot
    metrics_prebatch_meanaggr['method'] = 'Prebatch'
    metrics_prebatch_meanaggr.rename(columns={'recall': 'Recall', 'auc': 'AUC', 'f1': 'F1', 'accuracy': 'Accuracy', 'precision': 'Precision'}, inplace=True)
    
    metrics_postbatch_maxaggr['method'] = 'Postbatch'
    metrics_postbatch_maxaggr.rename(columns={'recall': 'Recall', 'auc': 'AUC', 'f1': 'F1', 'accuracy': 'Accuracy', 'precision': 'Precision'}, inplace=True)
    
    metrics_prebatch_gan_meanaggr['method'] = 'Prebatch GAN'
    metrics_prebatch_gan_meanaggr.rename(columns={'recall': 'Recall', 'auc': 'AUC', 'f1': 'F1', 'accuracy': 'Accuracy', 'precision': 'Precision'}, inplace=True)
    
    concat_metrics = pd.concat([metrics_prebatch_meanaggr, metrics_postbatch_maxaggr, metrics_prebatch_gan_meanaggr])
    melted_metrics = pd.melt(concat_metrics, id_vars=['method'], value_vars=['Recall', 'AUC', 'F1', 'Accuracy', 'Precision'])
    # replace 0.0 with None
    melted_metrics['value'] = melted_metrics['value'].replace(0.0, None)

    fig = px.box(melted_metrics, x="variable", y="value", color="method", color_discrete_map={
            "Prebatch": "#0047AB",
            "Postbatch": "#00A99D",
            "Prebatch GAN": "#6F00A8"
        },
        category_orders={'variable': ['recall','auc', 'f1','accuracy', 'precision']},
        width=600,
        height=400
        )
    fig.update_layout(
        yaxis_title='Performance',
        xaxis_title='Metric',
        font_family = "Moderne Computer",
        boxmode='group',
        legend_title_text='Method',
        boxgroupgap=0.2,
        boxgap=0.5,
        template='simple_white'
    )
    """for i in range(1, 5):
        fig.add_vline(x=i-0.5, line_width=1, line_dash="dash", line_color="black")"""
    fig.write_image('all_metrics_mixedaggr.pdf')

plot_all_metrics()
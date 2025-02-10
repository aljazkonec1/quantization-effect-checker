import os
import glob
import pandas as pd
import plotly.graph_objects as go

def create_scatter_plot(data_dir):
    all_dfs = []
    
    for file_path in glob.glob(os.path.join(data_dir, '*.csv')):
        # Read the CSV file.
        df = pd.read_csv(file_path)
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        df['Model'] = model_name
        all_dfs.append(df)
    
    data = pd.concat(all_dfs, ignore_index=True)
    metrics = ['CosSim', 'MSE', 'SAR', 'PearsonR', 'DKL']
    initial_metric = metrics[0]
    
    traces_data = {}
    models = data['Model'].unique()
    for model in models:
        df_model = data[data['Model'] == model]
        x_data = df_model['Layer'].tolist()
        metric_dict = {}
        for metric in metrics:
            metric_dict[metric] = df_model[metric].tolist()
        traces_data[model] = {'x': x_data, **metric_dict}
    
    fig = go.Figure()
    
    for model in models:
        fig.add_trace(go.Scatter(
            x=traces_data[model]['x'],
            y=traces_data[model][initial_metric],
            mode='markers',
            name=model,
            hovertemplate=f'Model: {model}<br>Layer: %{{x}}<br>{initial_metric}: %{{y}}<extra></extra>'
        ))
    
    buttons = []
    for metric in metrics:
        new_y = []
        new_hovertemplates = []
        for model in models:
            new_y.append(traces_data[model][metric])
            new_hovertemplates.append(f'Model: {model}<br>Layer: %{{x}}<br>{metric}: %{{y}}<extra></extra>')
            
        button = dict(
            label=metric,
            method="update",
            args=[
                {"y": new_y, "hovertemplate": new_hovertemplates},
                {"yaxis": {"title": metric}}
            ]
        )
        buttons.append(button)
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",         # This makes the buttons always visible.
                buttons=buttons,
                direction="right",      # Arrange buttons horizontally.
                showactive=True,
                x=0.5,                  # Center horizontally.
                xanchor="center",
                y=1.15,
                yanchor="top",
                pad={"r": 10, "t": 10}
            )
        ],
        xaxis_title="Layer",
        yaxis_title=initial_metric,
        title="Layer Performance Metrics by Model",
        hoverlabel=dict(
        font=dict(
            size=16
        )
    )
    )
    
    return fig

if __name__ == '__main__':
    data_directory = "results/similarities"
    fig = create_scatter_plot(data_directory)
    fig.write_html("visualizations/similarities.html", include_plotlyjs="cdn")
    fig.show()

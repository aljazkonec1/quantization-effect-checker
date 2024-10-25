import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_results_comparison_table(results_dir= "results"):
    results = os.listdir(results_dir)

    data = []
    for result in results:
        if not result.endswith(".json"):
            continue
        
        result_path = f"{results_dir}/{result}"
        results = json.load(open(result_path))
        model_name = results["info"]["model_name"]
        gt_path = results["info"]["gt_labels"]
        fps = results["info"]["FPS"]
        
        cocoGt = COCO(gt_path)
        cocoDt = COCO(result_path)
        
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        stats = cocoEval.stats
        
        confidences = []
        for res in results["annotations"]:
            confidences.append(res["score"])
        confidences = np.array(confidences)
            
        stats = [round(s, 4) for s in stats]
        
        res = {
            "model_name": model_name,
            "mAP@[IoU=0.50:0.95]": stats[0], 
            # "mAP@[IoU=0.50]": stats[1], 
            # "mAP@[IoU=0.75]": stats[2], 
            "AR@[IoU=0.50:0.95]": stats[8],
            "Avg_confidence": np.mean(confidences).round(4),
            "FPS": fps
        }
        data.append(res)

    df = pd.DataFrame(data)
    df["mAP-FPS score"] = (df["mAP@[IoU=0.50:0.95]"] * df["FPS"] / max(df["FPS"])).round(4)
    df = df.sort_values(by="mAP-FPS score", ascending=False)
    new_column_order = [
    "model_name",
    "mAP-FPS score",
    "mAP@[IoU=0.50:0.95]",
    "AR@[IoU=0.50:0.95]",
    "Avg_confidence",
    "FPS"
    ]

    df = df[new_column_order]
    df.to_csv(f"{results_dir}/results.csv", index=False)
    return df

def create_plotly_comparison_graph(df_full: pd.DataFrame, results_df = pd.DataFrame) -> go.Figure:
    df_pivot_cycles = df_full.pivot_table(index='layer_id', columns='model_name', values='time_cycles', fill_value=0)
    df_pivot_cycles = df_pivot_cycles.astype(int)
    df_pivot_cycles = df_pivot_cycles / 1000
    df_pivot_percentage = df_full.pivot_table(index='layer_id', columns='model_name', values='Percentage_of_Total_Time', fill_value=0)
    total_cycles = df_full.groupby('model_name')['time_cycles'].sum()
    
    fig = go.Figure()

    # --- Add Time Cycles per Layer Traces ---
    for model in df_pivot_cycles.columns:
        y_values = df_pivot_cycles[model]
        layer_ids = df_pivot_cycles.index
        layer_names = df_full[df_full['model_name'].str.strip() == model].set_index('layer_id').loc[layer_ids]['layer_name']
        layer_names = layer_names.reindex(layer_ids)
        
        hover_text = [
            f"{model.strip()} : {y_value} cycles<br>{layer_name}"
            for layer_name, y_value in zip(layer_names, y_values)
        ]
        
        fig.add_trace(go.Bar(
            x=layer_ids,
            y=y_values,
            name=model.strip(),
            hovertext=hover_text,
            hoverinfo="text",
            visible=True  # Initially visible
        ))
    # --- Add Percentage of Total Time Traces (Hidden by Default) ---
    for model in df_pivot_percentage.columns:
        fig.add_trace(go.Bar(
            x=df_pivot_percentage.index,
            y=df_pivot_percentage[model],
            name=model.strip(),
            hovertext=df_full[df_full['model_name'].str.strip() == model]['layer_name'],
            hoverinfo="text+y",
            visible=False
        ))
    
    for model in total_cycles.index:
        fig.add_trace(go.Bar(
            x=[model],
            y=[total_cycles[model]],
            name=model.strip(),
            hoverinfo="x +y",
            visible=False
        ))
    
    for model in results_df['model_name']:
        fig.add_trace(go.Scatter(
            x=results_df[results_df['model_name'] == model]['FPS'],
            y=results_df[results_df['model_name'] == model]['mAP@[IoU=0.50:0.95]'],
            hovertext=str(results_df[results_df['model_name'].str.strip() == model]['mAP-FPS score'].values[0]) + ' mAP-FPS Score' + '<br>' + model + '<br>',
            hoverinfo="text",
            mode='markers',
            marker=dict(size=14),
            name=model,
            visible=False
        ))

    
    # --- Calculate Number of Traces ---
    N1 = len(df_pivot_cycles.columns)  # Time cycles traces
    N2 = len(df_pivot_percentage.columns)  # Percentage traces
    N3 = len(total_cycles.index)  # Total cycles per model
    N4 = len(results_df['model_name'])

    # --- Update Layout with Buttons ---
    fig.update_layout(
        title="Per layer comparison of models",
        xaxis=dict(title="Layer ID"),
        yaxis=dict(title="Time (cycles)"),
        barmode='group',
        hovermode='x',
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.15,
                showactive=True,
                buttons=list([
                    # Button for Time (cycles)
                    dict(
                        label="Time (cycles)",
                        method="update",
                        args=[
                            {"visible": [True] * N1 + [False] * (N2 + N3 + N4)},
                            {"title": "Per layer comparison of models",
                             "xaxis": {"title": "Layer ID"},
                             "yaxis": {"title": "Time (cycles)", "tickformat": ""},
                             "barmode": "group",
                             "hovermode": "x",
                             "showlegend": True}
                        ]
                    ),
                    # Button for Percentage of Total Time
                    dict(
                        label="Percentage of Total Time",
                        method="update",
                        args=[
                            {"visible": [False] * (N1) + [True] * N2 + [False] * (N4 + N3)},
                            {"title": "Percentage of Total Time per Layer",
                             "xaxis": {"title": "Layer ID"},
                             "yaxis": {"title": "% of Total Time", "tickformat": ".2%"},
                             "barmode": "group",
                             "hovermode": "x",
                             "showlegend": True}
                        ]
                    ),
                    # Button for General Information
                    dict(
                        label="General Information",
                        method="update",
                        args=[
                            {"visible": [False] * (N1 + N2) + [True] * N3 + [False] * N4},
                            {"title": "Total Cycles per Model",
                             "xaxis": {"title": "Model"},
                             "yaxis": {"title": "Total Cycles", "tickformat": ""},
                             "barmode": "group",
                             "hovermode": "closest",
                             "showlegend": False}
                        ]
                    ),
                    # Button for FPS vs mAP@[IoU=0.50:0.95] Scatter Plot
                    dict(
                        label="FPS vs mAP@[IoU=0.50:0.95]",
                        method="update",
                        args=[
                            {"visible": [False] * (N1 + N2 + N3) + [True] * N4},
                            {"title": "FPS vs mAP@[IoU=0.50:0.95]",
                             "xaxis": {"title": "FPS"},
                             "yaxis": {"title": "mAP@[IoU=0.50:0.95]"},
                             "barmode": "group",
                             "hovermode": "closest",
                             "showlegend": True}
                        ]
                    )
                ])
            )
        ]
    )

    return fig


def create_per_layer_analysis(results_df: pd.DataFrame, results_dir= "results") -> go.Figure:
    df = pd.DataFrame()
    for result in os.listdir(results_dir):
        if not result.endswith(".json"):
            continue
        
        result_path = f"{results_dir}/{result}"
        results = json.load(open(result_path))
        value = results["info"].get("per_layer_analysis", None)
        
        if value is None:
            continue
        
        #concat at the end
        per_layer_df = pd.read_csv(value)
        per_layer_df["model_name"] = results["info"]["model_name"]
        per_layer_df["layer_name"] = per_layer_df["layer_name"].apply(lambda x: x.split(":")[0])
        per_layer_df["time_cycles"] = per_layer_df["time_mean"]
        per_layer_df.drop(columns=["message", "time_mean", "unit"], inplace=True)
        
        df = pd.concat([df, per_layer_df])
    
    model_with_most_layers = df['model_name'].value_counts().idxmax()
    df_complete = df[df['model_name'] == model_with_most_layers][["layer_id", "layer_name"]].copy()

    df_full = pd.DataFrame()
    for model in df.groupby('model_name'):
        model = model[1][['layer_name', 'model_name', 'time_cycles', 'time_std', 'Percentage_of_Total_Time']].merge(df_complete, on='layer_name', how='right')
        model["model_name"] = model["model_name"].ffill()
        model = model.fillna(0)
        df_full = pd.concat([df_full, model])

    df_full.columns = df_full.columns.str.strip()
    # df_full.rename(columns={'Time (cycles)': 'time_cycles', 'Time Std': 'time_std', 'Layer Name': 'layer_name', 'Layer Id': 'layer_id'}, inplace=True)
    df_full['layer_name'] = df_full['layer_name'].apply(lambda x: x.split(' (cycles)')[0])
    df_full['time_cycles'] = df_full['time_cycles'].astype(int)
    df_full['time_std'] = df_full['time_std'].astype(int)   
    

    df_pivot_cycles = df_full.pivot_table(index='layer_id', columns='model_name', values='time_cycles', fill_value=0)
    layer_info = df_full[['layer_id', 'layer_name']].drop_duplicates()
    
    df_table = df_pivot_cycles.reset_index().merge(layer_info, on='layer_id')
    cols = ['layer_name'] + [col for col in df_table.columns if col not in ['layer_id', 'layer_name']]
    df_table = df_table[cols]
    df_table.to_csv(f"{results_dir}/per_layer_comparison.csv", index=False)
    
    
    fig = create_plotly_comparison_graph(df_full, results_df)

    return fig

if __name__ == "__main__":
    # df = create_results_comparison_table()
    df = pd.read_csv("results/results.csv")
    fig = create_per_layer_analysis(df )
    
    fig.show()
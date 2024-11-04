import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
import fiftyone as fo
from utils import update_menus

warnings.filterwarnings(
    "ignore", 
    message="Resizing predicted mask with shape", 
    category=UserWarning
)

def detection_statistics(results_dir= "results"):
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

def segmentation_statistics(results_dir: str = "results", gt_dir = "data/test") -> pd.DataFrame:
    all_results = []
    dataset = fo.Dataset.from_dir( # ground truth
        dataset_type=fo.types.ImageSegmentationDirectory,
        data_path= gt_dir + "/data",
        labels_path=gt_dir + "/gt",
        label_field="ground_truth"
    )
    
    all_prediction_results = os.listdir(results_dir)
    for prediction in all_prediction_results:
        if not prediction.endswith(".json"):
            continue
        
        info_json = json.load(open(f"{results_dir}/{prediction}"))["info"]
        predictions_dir = info_json["predictions_dir"]
        
        predictions_dataset = fo.Dataset.from_dir( # predictions
            dataset_type=fo.types.ImageSegmentationDirectory,
            data_path=gt_dir + "/data",
            labels_path=predictions_dir,
            label_field="predictions"
        )
        dataset.merge_samples(predictions_dataset)
    
        # # Evaluate segmentation results
        results = fo.utils.eval.segmentation.evaluate_segmentations(
            dataset,
            pred_field="predictions",
            gt_field="ground_truth",
            method="simple",
            eval_key="eval",
        )
        results = results.metrics()
        
        results["model_name"] = info_json["model_name"]
        results["FPS"] = info_json["FPS"]
        all_results.append(results)
        
        dataset.delete_sample_field("predictions")
    
    df = pd.DataFrame(all_results)
    df = df.drop(columns=["support"])
    df = df.round(4)
    df = df.sort_values(by="fscore", ascending=False)
    new_column_order = [
        "model_name",
        "fscore",
        "precision",
        "recall",
        "accuracy",
        "FPS"
        ]
    df = df[new_column_order]
    df.to_csv(f"{results_dir}/results.csv", index=False)
    
    return df

def create_plotly_comparison_graph(df_full: pd.DataFrame, results_dir: str = "results", prediction_type: str = "detection") -> go.Figure:
    df_pivot_cycles = df_full.pivot_table(index='layer_id', columns='model_name', values='time_cycles', fill_value=0)
    df_pivot_cycles = df_pivot_cycles.astype(int)
    df_pivot_cycles = df_pivot_cycles
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
        y_values = df_pivot_percentage[model]
        layer_ids = df_pivot_percentage.index
        layer_names = df_full[df_full['model_name'].str.strip() == model].set_index('layer_id').loc[layer_ids]['layer_name']
        layer_names = layer_names.reindex(layer_ids)
        
        hover_text = [
            f"{model.strip()} : {y_value*100} %<br>{layer_name}"
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
        
    for model in total_cycles.index:
        fig.add_trace(go.Bar(
            x=[model],
            y=[total_cycles[model]],
            name=model.strip(),
            hoverinfo="x +y",
            visible=False
        ))
    
    results_df = pd.read_csv(f"{results_dir}/results.csv")
    
    if prediction_type == "segmentation":
        for model in results_df['model_name']:
            fig.add_trace(go.Scatter(
                x=results_df[results_df['model_name'] == model]['FPS'],
                y=results_df[results_df['model_name'] == model]['fscore'],
                hovertext=str(results_df[results_df['model_name'].str.strip() == model]['fscore'].values[0]) + ' F1 Score' + '<br>' + model + '<br>',
                hoverinfo="text",
                mode='markers',
                marker=dict(size=14),
                name=model,
                visible=False
            ))
    
    else: # detection
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

    buttons = update_menus(N1, N2, N3, N4, prediction_type)
    
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
                buttons=buttons
            )
        ]
    )

    return fig

def create_per_layer_analysis(results_dir: str = "results", prediction_type: str = "detection") -> go.Figure:
    
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
    
    if "segmentation" in prediction_type:
        segmentation_statistics(results_dir, "data/test")
    else:
        detection_statistics(results_dir)
            
    
    fig = create_plotly_comparison_graph(df_full, results_dir, prediction_type)

    return fig


    
if __name__ == "__main__":
    # df = create_results_comparison_table()
    # df = pd.read_csv("results/results.csv")
    fig = create_per_layer_analysis("results", "segmentation")
    
    fig.show()
    
    # segmentation_statistics("data/test","results" )
    
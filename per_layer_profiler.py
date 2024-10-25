import pandas as pd

def per_layer_profiles(layers_stats_csv, output_path = "models_dlc"):
    df = pd.read_csv(layers_stats_csv, index_col=False)
    other_stats = df[df['Layer Id'].isna()]
    filtered_df = df[df['Layer Id'].notna()]
    layer_stats = filtered_df.groupby('Layer Id').agg({
        'Time': ['mean', 'std'],  # Calculate mean and standard deviation of Time
        'Message': 'first',       # Retain the first Message for each Layer Id (or use another function like 'unique')
        'Layer Name': 'first',    # Retain the first Layer Name (or use 'unique' if needed)
        'Unit of Measurement': 'first',  # Retain the first Unit of Measurement
    }).reset_index()

    # Flatten the column names
    layer_stats.columns = ['Layer Id', 'Time Mean', 'Time Std', 'Message', 'Layer Name', 'Unit of Measurement']
    layer_stats['Time Mean'] = layer_stats['Time Mean'].round(0)
    layer_stats['Time Std'] = layer_stats['Time Std'].round(0)
    layer_stats['Time Mean'] = layer_stats['Time Mean'].astype(int)
    layer_stats['Time Std'] = layer_stats['Time Std'].astype(int)
    layer_stats['Layer Id'] = layer_stats['Layer Id'].astype(int)
    total_time = layer_stats['Time Mean'].sum()
    layer_stats['Percentage_of_Total_Time'] = ((layer_stats['Time Mean'] / total_time)).round(4)
    
    layer_stats.rename(columns={'Layer Id': 'layer_id', 'Time Mean': 'time_mean', 'Time Std': 'time_std', 'Message': 'message', 'Layer Name': 'layer_name', 'Unit of Measurement': 'unit'}, inplace=True)
    layer_stats.to_csv(output_path, index=False)
    

if __name__ == "__main__":
    # per_layer_profiles("models_dlc/yolov6n-per-channel-quant/output_log_per-channel-quant.csv")
    per_layer_profiles("models_dlc/yolov6n-base-quant/layer_stats.csv", "models_dlc/yolov6n-base-quant/layer_stats.csv")
    per_layer_profiles("models_dlc/yolov6n-per-channel-conv-transpose/layer_stats.csv", "models_dlc/yolov6n-per-channel-conv-transpose/layer_stats.csv")    
    per_layer_profiles("models_dlc/yolov6n-per-channel-fused/layer_stats.csv", "models_dlc/yolov6n-per-channel-fused/layer_stats.csv")    
    per_layer_profiles("models_dlc/yolov6n-per-channel-reordered/layer_stats.csv", "models_dlc/yolov6n-per-channel-reordered/layer_stats.csv")    
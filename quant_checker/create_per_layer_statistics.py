import numpy as np
import os
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy


def process_single_output(raw_file_path: str, shape: tuple, dtype: np.dtype = np.float32) -> np.ndarray:
    with open(raw_file_path, "rb") as f:
        raw_data = f.read() 

    output = np.frombuffer(raw_data, dtype=dtype)
    output = output.reshape(shape)
    return output
    
def kl_divergence(data1, data2):
    
    p, _ = np.histogram(data1, density=True)
    q, _ = np.histogram(data2, density=True)
    
    p = np.where(p == 0, 1e-10, p)  # Avoid log(0) issue
    q = np.where(q == 0, 1e-10, q)
    return entropy(p, q)
    
def strip_underscores(dir_name: str):
    
    for file in os.listdir(dir_name):
        new_file = file.strip("_")
        os.rename(os.path.join(dir_name, file), os.path.join(dir_name, new_file))
    
def compare_outputs(model_name, onnx_config_path, dlc_config_path, graph_info_path):
    """
    model_name: str
        Name of the model. Very imporatnt as the script assumes onnx outputs are in "onnx_models/{model_name}/output" and dlc outputs are in "models_dlc/{model_name}/output"
    onnx_config: str
        The path to the json file that contains info on onnx layer output shapes.
    dlc_config: str
        The path to json file that contains paths to dlc output files for every layer.
    graph_info: str
        The path to the json file that contains the dlc graph output shapes.
    """
    
    with open( onnx_config_path, "r") as f:
        onnx_config = json.load(f)
    
    with open(dlc_config_path, "r") as f:
        dlc_config = json.load(f)
    
    with open(graph_info_path, "r") as f:
        graph_info = json.load(f)
    
    similarities = []
    for layer_name, onnx_layer_shape in onnx_config.items():
        if dlc_config.get(layer_name) is None or graph_info.get(layer_name) is None:
            continue
        
        dlc_layer_shape = graph_info[layer_name]
        fw_type = np.float32
        
        onnx_output = process_single_output(os.path.join("models_onnx", model_name, "output", f"{layer_name}.raw"), onnx_layer_shape, fw_type)
        dlc_output = process_single_output(os.path.join("models_dlc", model_name, dlc_config[layer_name]), dlc_layer_shape, fw_type)
        
        if dlc_output.shape != onnx_output.shape:
            dlc_output = np.transpose(dlc_output, [0, 3, 1, 2])
        
        sim = cosine_similarity(onnx_output.flatten().reshape(1, -1), dlc_output.flatten().reshape(1, -1))[0][0]
        mse = mean_squared_error(onnx_output.flatten(), dlc_output.flatten())
        sar = np.average((onnx_output >= 0) == (dlc_output >= 0)) #sign agreement rate
        
        pearsonr = np.corrcoef(onnx_output.flatten(), dlc_output.flatten())[0][1]        
        dkl = kl_divergence(onnx_output.flatten(), dlc_output.flatten())
            
        similarities.append((layer_name, sim, mse, sar, pearsonr, dkl))
        
    
    sim_df = pd.DataFrame(similarities, columns=["Layer","CosSim", "MSE", "SAR", "PearsonR", "DKL"])

    sim_df.to_csv(f"results/similarities/{model_name}.csv", index=False)
    sim_df.to_csv(f"models_dlc/{model_name}/similarity.csv", index=False)
import os
import numpy as np
import shutil
from rich.console import Console
import onnx
import onnxruntime as rt
import cv2
import json
console = Console()

def create_raw_file_for_test(img: np.ndarray, dir_name: str):
    """
    Create a raw file from an image for per layer analysis.
    The data is saved in data_raw/{dir_name}
    """ 
    
    output_dir = f"data_raw/{dir_name}"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    img = img.astype(np.float32)
    img_raw = img.tobytes()
    raw_file_name = os.path.join(output_dir,"test.raw")
    
    with open(raw_file_name, "wb") as f:
        f.write(img_raw)
    
    input_list = os.path.join(dir_name, "test.raw")
    with open(os.path.join(output_dir, "inputs_raw.txt"), "w") as fp:
        fp.write(input_list)
    
    console.print(f"[bold green] Raw file created at {output_dir} [/bold green]")
    
def add_outputs_to_all_layers(onnx_file_path: str) -> str:
    if os.path.exists(onnx_file_path.replace(".onnx", "-all-layers.onnx")):
        return onnx_file_path.replace(".onnx", "-all-layers.onnx")

    model = onnx.load(onnx_file_path)
    onnx.checker.check_model(model)
    graph = model.graph
    orig_graph_output = list(graph.output)
    
    del graph.output[:]
    
    for node in graph.node:
        for output in node.output:
            if output not in [o.name for o in graph.output]:
                value_info = onnx.helper.ValueInfoProto()
                value_info.name = output
                graph.output.append(value_info)
    
    for output in orig_graph_output:
        graph.output.append(output)
    all_output_name = onnx_file_path.replace(".onnx", "-all-layers.onnx")
    onnx.save(model, all_output_name)
    console.print(f"[bold] Created onnx file with all layers at {all_output_name} [/bold]")
    
    return all_output_name

def run_onnx_model(onnx_model_path, model_name, img_path):
    """
    """
    console.print(f"[bold cyan]Running onnx model...[/bold cyan]")
    
    model_dir = "models_onnx/" + model_name
    os.makedirs(model_dir, exist_ok=True)
    shutil.copyfile(onnx_model_path, f"{model_dir}/{model_name}.onnx")
    
    if os.path.exists(model_dir + "/output"):
        shutil.rmtree(model_dir + "/output")
    os.makedirs(model_dir + "/output")
    
    console.print(f"[bold] Creating onnx output files for img {img_path}... [/bold]")
    onnx_all_layers_path = add_outputs_to_all_layers(f"{model_dir}/{model_name}.onnx")
    
    session = rt.InferenceSession(onnx_all_layers_path)
    layer_names = session.get_outputs()
    layer_names = [l.name for l in layer_names]
    input_shape = [ x.shape for x in session.get_inputs()][0][2:]
    input_shape = input_shape[::-1]
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, input_shape ) # NCHW format is assumed by default
    
    create_raw_file_for_test(img, model_name)
    
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0).astype(np.float32)
    layer_outputs = session.run(layer_names, {session.get_inputs()[0].name: img})
    
    config_json = {}
    for layer_name, layer_output in zip(layer_names, layer_outputs):
        ln = layer_name.replace("..", ".")
        ln = layer_name.replace("__", "_")
        ln = layer_name.replace(".", "_")
        ln = ln.replace("/", "_")
        ln = ln.strip("_")
        ln = ln.replace("_output_0", "")
        
        output_file_path = os.path.join(model_dir, "output", f"{ln}.raw")
        shape = [int(d) for d in layer_output.shape]
        config_json[ln] = shape
          
        raw_output = layer_output.tobytes()
                
        with open(output_file_path, "wb") as f:
            f.write(raw_output)
            
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config_json, f)
        
    console.print(f"[green] Created onnx output files at {model_dir}. [/green]")

    return model_dir
    
import os
import onnx
import onnxruntime
import cv2
import numpy as np
from postprocess_functions import process_yolov6_outputs, process_segmentation_outputs
from quant_checker.utils.utils import resize_and_pad
import json
from onnx import TensorProto
import shutil
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
console = Console()

def run_models_onnx(models_dir="models_onnx", 
                    images_dir = "data/test",
                    save_dir = "raw_outputs",
                    postprocess_function = process_yolov6_outputs):
    """
    Parameters
    -----------
    models_dir: str
        Path to directory containing the onnx models.
    labels_dir: str
        Path to directory containing the labels.json file and coresponding images
    postprocess_function: a postprocessing function that converts model output int compatible COCO-style annotation
    """
    console.print(Panel("[bold purple]Running ONNX models... [/bold purple]"))
    
    models = os.listdir(models_dir)

    for model in models:
        if not model.endswith(".onnx"):
            continue
        console.print(f"Running model: [bold blue]{model}[/bold blue]")
        model_name = model.replace(".", "-")
        onnx_model = onnx.load(f"{models_dir}/{model}")
        
        input_tensor = onnx_model.graph.input[0]
        input_name = input_tensor.name
        elem_type = input_tensor.type.tensor_type.elem_type
        if elem_type == TensorProto.FLOAT16:
            elem_type = np.float16
        else:
            elem_type = np.float32
        input_shape = tuple([dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim])
        model_h, model_w = input_shape[2], input_shape[3]
        
        output_tensor = onnx_model.graph.output
        output_names = [output.name for output in output_tensor]
        
        onnx_model = onnxruntime.InferenceSession(f"models_onnx/{model}")
        
        images = os.listdir(images_dir + "/data")
        os.makedirs(f"processed_outputs/{model_name}", exist_ok=True)
        
        for img_name in images:
            
            img_path = f"{images_dir}/data/{img_name}"
            img = cv2.imread(img_path)
            
            img = cv2.resize(img, (513, 513))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
            img = img.transpose(2, 0, 1)
            img = img / 255
            img = np.expand_dims(img, axis=0)
            
            outputs = onnx_model.run(output_names, {input_name: img.astype(elem_type)})
            
            segmentation, probabilities = postprocess_function(outputs[0])
            # segmentation = cv2.cvtColor(segmentation, cv2.COLOR_GRAY2BGR)
            save_name = img_name.replace(".jpg", ".png")
            cv2.imwrite(f"processed_outputs/{model_name}/{save_name}", segmentation)
        
        info = {}
        info["model_name"] = model_name
        info["gt_labels"] = f"{images_dir}/data"
        info["FPS"] = -1
        info["raw_outputs_dir"] = f"{save_dir}/{model_name}"
        info["categories"] = json.load(open(f"{images_dir}/categories.json"))
        info["predictions_dir"] = f"processed_outputs/{model_name}"

        json_file = {"info": info}
        with open(f"results/{model_name}.json", "w") as f:
            json.dump(json_file, f)
        
def run_models_dlc(models_dir="models_dlc",
                   postprocess_function = process_yolov6_outputs):
    """
    Parameters
    -----------
    models_dir: str
        Path to directory containing the dlc models.
    postprocess_function: a postprocessing function that converts model output int compatible COCO-style annotation
    """
    console.print(Panel("[bold purple]Running DLC models... [/bold purple]"))
    
    models_dir = [os.path.join(models_dir, model) for model in os.listdir(models_dir)]
    
    for model_dir in models_dir:
        if not os.path.isdir(model_dir):
            continue
        if not os.path.exists(f"{model_dir}/info.json"):
            continue
        model = model_dir.split("/")[-1]
        console.print(f"Running model: [bold blue]{model}[/bold blue]")
        info = json.load(open(f"{model_dir}/info.json"))
        
        file = []
        with open(info["text_file_path"], "r") as f:
            file = f.readlines()
        
        raw_outputs_dir = info["raw_outputs_dir"]
        
        raw_outputs = os.listdir(raw_outputs_dir)
        raw_outputs = [os.path.join(raw_outputs_dir,entry) for entry in raw_outputs if os.path.isdir(os.path.join(raw_outputs_dir, entry))]
        raw_outputs = sorted(raw_outputs, key=lambda x: int(x.split("_")[-1]))
    
        dequantize_info = info["dequantize_info"]
        output_names = info["output_names"]
        model_name = info["model_name"]
        
        if os.path.exists(f"processed_outputs/{model_name}"):
            shutil.rmtree(f"processed_outputs/{model_name}")
        os.makedirs(f"processed_outputs/{model_name}", exist_ok=True)
        
        for i, image_name in enumerate(file):
            image_name = image_name.strip().split("/")[-1].split(".")[0]
            
            outputs = {}
            for output_name in os.listdir(raw_outputs[i]):
                output_path = os.path.join(raw_outputs[i], output_name)
                
                output_name = output_name.split(".")[0] # outputX_yolov6r2
                output_dequantize_info = dequantize_info[output_name]

                with open(output_path, "rb") as f:
                    output = f.read()
                if "Float_16" in output_dequantize_info["type"]:
                    data_type = np.float16
                elif "Float_32" in output_dequantize_info["type"]:
                    data_type = np.float32
                elif "uFxp_16" in output_dequantize_info["type"]:
                    data_type = np.uint16
                else:
                    data_type = np.uint8
                
                output = np.frombuffer(output, dtype=data_type)
                output = output.reshape(tuple(output_dequantize_info["shape"]))
                # if "uFxp_" in output_dequantize_info["type"]:
                output = output_dequantize_info["scale"] * output + output_dequantize_info["min"]
                
                # if "uFxp_" not in output_dequantize_info["type"]:
                #     print("FP 16")
                #     print(output_name)
                #     print(output)
                #     break
                output = output.transpose(0, 3, 1, 2)
                outputs[output_name] = output
            
            outputs_list =[]
            for i in range(len(output_names)):
                outputs_list.append(outputs[output_names[i]])
            segmentation, probabilities = postprocess_function(outputs_list[0])

            # save_name = image_name.replace(".jpg", ".png")
            cv2.imwrite(f"processed_outputs/{model_name}/{image_name}.png", segmentation)
        model_name = info["model_name"]
        
        info["predictions_dir"] = f"processed_outputs/{model_name}"
        json_file = {"info": info}

        with open(f"results/{model_name}.json", "w") as f:
            json.dump(json_file, f)

if __name__ == "__main__":
    run_models_onnx(postprocess_function=process_segmentation_outputs)
    # run_models_dlc(postprocess_function=process_segmentation_outputs)

    
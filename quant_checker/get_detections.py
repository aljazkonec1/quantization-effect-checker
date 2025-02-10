import os
import onnx
import onnxruntime
import cv2
import numpy as np
from postprocess_functions import process_yolov6_outputs
from quant_checker.util_functions import resize_and_pad
import json
from onnx import TensorProto

def run_models_onnx(models_dir="models_onnx", 
                    labels_dir="data/test",
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
    models = os.listdir(models_dir)
    labels = json.load(open(f"{labels_dir}/labels.json"))
    images = labels["images"]
    classes = [ c["id"] for c in labels["categories"]]
    
    for model in models:
        if not model.endswith(".onnx"):
            continue
        print("Running model: ", model)
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
        
        detection_id = 0 
        detections = []
        for img_info in images:
            img_name = img_info["file_name"]
            img_path = f"{labels_dir}/data/{img_name}"
            img_id = img_info["id"]
            
            width = img_info["width"]
            height = img_info["height"]
            
            img = cv2.imread(img_path)
            img = resize_and_pad(img, target_size=(model_w, model_h))
            img = img.transpose(2, 0, 1)
            img = img / 255
            img = np.expand_dims(img, axis=0)
            
            outputs = onnx_model.run(output_names, {input_name: img.astype(elem_type)})
            
            boxes = postprocess_function(outputs)

            for box in boxes:
                x1, y1, x2, y2, conf, cls_score = box[:6].cpu().numpy()  # Assuming 'boxes' is on a GPU tensor
                x1, y1, x2, y2 = int(x1*width/model_w), int(y1*height/model_h), int(x2*width/model_w), int(y2*height/model_h)
                class_idx = int(classes[int(cls_score)])
                
                w = x2 - x1
                h = y2 - y1
            
                #create json annotation
                detection = {
                    "image_id": img_id,
                    "category_id": class_idx,
                    "bbox": [x1, y1, w, h],
                    "score": float(conf),
                    "id": detection_id,  # Add a unique ID
                    "area" : w * h
                }
                detection_id += 1
                detections.append(detection)
        
        annotations = {"annotations": detections}
        
        info = {}
        info["model_name"] = model.replace(".", "-")
        info["gt_labels"] = f"{labels_dir}/labels.json"
        info["FPS"] = -1
        annotations["info"] = info
        annotations["images"] = images
        annotations["categories"] = labels["categories"]
        with open(f"results/{model.replace(".", "-")}.json", "w") as f:
            json.dump(annotations, f)
        
def run_models_dlc(models_dir="models_dlc",
                   postprocess_function = process_yolov6_outputs):
    """
    Parameters
    -----------
    models_dir: str
        Path to directory containing the dlc models.
    postprocess_function: a postprocessing function that converts model output int compatible COCO-style annotation
    """
    
    
    models_dir = [os.path.join(models_dir, model) for model in os.listdir(models_dir)]
    

    for model_dir in models_dir:
        if not os.path.isdir(model_dir):
            continue
        if not os.path.exists(f"{model_dir}/info.json"):
            continue
        print("Running model: ", model_dir.split("/")[-1])
        info = json.load(open(f"{model_dir}/info.json"))
        
        gt_labels = json.load(open(info["gt_labels"]))
        images = gt_labels["images"]
        classes = [ c["id"] for c in gt_labels["categories"]]

        file = []
        with open(info["text_file_path"], "r") as f:
            file = f.readlines()
        
        raw_outputs_dir = info["raw_outputs_dir"]
        raw_outputs = os.listdir(raw_outputs_dir)
        raw_outputs = [os.path.join(raw_outputs_dir,entry) for entry in raw_outputs if os.path.isdir(os.path.join(raw_outputs_dir, entry))]
        raw_outputs = sorted(raw_outputs, key=lambda x: int(x.split("_")[-1]))
    
        dequantize_info = info["dequantize_info"]
        output_names = info["output_names"]
        model_h = info["model_h"]
        model_w = info["model_w"]
        
        detection_id = 0
        detections = []
        
        for i, image_name in enumerate(file):
            image_name = image_name.strip().split("/")[-1].split(".")[0]
            image_info = [img for img in images if image_name in img["file_name"]][0]
            width = image_info["width"]
            height = image_info["height"]
            img_id = image_info["id"]
            img_name = image_info["file_name"]
            
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
            
            boxes = postprocess_function(outputs_list)
            for box in boxes:
                x1, y1, x2, y2, conf, cls_score = box[:6].cpu().numpy()  # Assuming 'boxes' is on a GPU tensor
                x1, y1, x2, y2 = int(x1*width/model_w), int(y1*height/model_h), int(x2*width/model_w), int(y2*height/model_h)
                
                class_idx = int(classes[int(cls_score)])
                
                w = x2 - x1
                h = y2 - y1
                
                #create json annotation
                detection = {
                    "image_id": img_id,
                    "category_id": class_idx,
                    "bbox": [x1, y1, w, h],
                    "score": float(conf),
                    "id": detection_id,  # Add a unique ID
                    "area" : w * h
                }
                detection_id += 1
                detections.append(detection)
            
        annotations = {"annotations": detections}
        annotations["info"] = info
        annotations["images"] = images
        annotations["categories"] = gt_labels["categories"]
        with open(f"results/{info["model_name"]}.json", "w") as f:
            json.dump(annotations, f)
     
if __name__ == "__main__":
    print("Running onnx models ...")
    run_models_onnx()
    print("Running dlc models ...")
    run_models_dlc()

    
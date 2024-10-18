import os
import cv2
import json
import math
import matplotlib.pyplot as plt

def draw_boxes(img, boxes, detected_class_names, color = (0,255,0)):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls_score = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        name = detected_class_names[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{cls_score}: {name}, score: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return img
    

def visualize_predictions(results_dir="results",
                    gt_dir="data/test"):
    """
    Visualized the predictions that are saved as .json file in the results_dir directory.
    
    Parameters
    ----------
    results_dir: str
        Path to the directory where the results are stored.
    gt_dir: str
        Path to the directory where the ground truth data is stored.
    """
    
    dlc_models = os.listdir(results_dir) 
    labels = json.load(open(f"{gt_dir}/labels.json"))
    images = labels["images"]
    classes = [ c["id"] for c in labels["categories"]]
    class_names = {c["id"]: c["name"] for c in labels["categories"]}

    n_models = len(dlc_models)
    plt_subplot_shape = (2, math.ceil(n_models/2))
    
    for img_info in images:
        img_name = img_info["file_name"]
        img_id = img_info["id"]
        
        fig, axes = plt.subplots(*plt_subplot_shape, figsize=(20, 10))
        axes = axes.T.flatten()
        fig.canvas.manager.set_window_title(img_id)
        
        img_path = f"{gt_dir}/data/{img_name}"
        img = cv2.imread(img_path)
        orig_img = img.copy()
        
        gts = [ann for ann in labels["annotations"] if ann["image_id"] == img_id]
        boxes = []
        gt_class_names = []
        for gt in gts:
            x, y, w, h = gt["bbox"]
            x2, y2 = int(x + w), int(y + h)
            boxes.append([int(x), int(y), x2, y2, 1, gt["category_id"]])
            gt_class_names.append(class_names[gt["category_id"]])
        img = draw_boxes(img, boxes, gt_class_names, color=(0,255,0))
        
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')

        img = orig_img.copy()
        i = 1
        
        for model in dlc_models:
            if not model.endswith(".json"):
                continue
            # print(model)
            
            predictions = json.load(open(f"{results_dir}/{model}"))
            info = predictions["info"]
            predicted_boxes = predictions["annotations"]
            predicted_boxes = [box for box in predicted_boxes if box["image_id"] == img_id]
            model_classes = predictions["categories"]
            model_classes = {c["id"]: c["name"] for c in model_classes}
            
            boxes = []
            predicted_class_names = [] 
            for box in predicted_boxes:
                x, y, w, h = box["bbox"]
                x2, y2 = int(x + w), int(y + h)
                boxes.append([int(x), int(y), x2, y2, box["score"], box["category_id"]])
                predicted_class_names.append(model_classes[box["category_id"]])
            
            img = draw_boxes(img, boxes, predicted_class_names, color=(255,255,0))
            
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"{model.split('.')[0]}")
            axes[i].axis('off')
            i = i + 1
            img = orig_img.copy()
        
        plt.show()
        
        plt.close(fig)
            
     
if __name__ == "__main__":
    visualize_predictions()
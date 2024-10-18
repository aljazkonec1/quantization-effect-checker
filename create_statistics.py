import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
import pandas as pd

results = os.listdir("results")
metrics = [
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]'
]

data = []
i = 0
for result in results:
    if not result.endswith(".json"):
        continue
    
    result_path = f"results/{result}"
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
         "mAP@[IoU=0.50]": stats[1], 
         "mAP@[IoU=0.75]": stats[2], 
        "AR@[IoU=0.50:0.95]": stats[8],
        "Avg_confidence": np.mean(confidences).round(4),
        "FPS": fps
    }
    data.append(res)

df = pd.DataFrame(data)
df.to_csv("results/results.csv", index=False)
    
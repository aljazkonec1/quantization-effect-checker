import numpy as np
import torch
import luxonis_ml
from luxonis_train.utils.boundingbox import (
    anchors_for_fpn_features,
    dist2bbox,
    non_max_suppression,
)
def process_to_bbox(
            output,
            conf_thres = 0.25
            
        ):
    # HARD CODED PARAMS
    grid_cell_size = 5
    grid_cell_offset = 0.5
    n_classes = 80
    iou_thres = 0.5
    max_det = 300
    stride = torch.tensor([8, 16, 32])

    """Performs post-processing of the output and returns bboxs after NMS."""
    features, cls_score_list, reg_dist_list = output
    _, anchor_points, _, stride_tensor = anchors_for_fpn_features(
        features,
        stride,
        grid_cell_size,
        grid_cell_offset,
        multiply_with_stride=False,
    )
    pred_bboxes = dist2bbox(reg_dist_list, anchor_points, out_format="xyxy")
    
    pred_bboxes *= stride_tensor
    output_merged = torch.cat(
        [
            pred_bboxes,
            torch.ones(
                (features[-1].shape[0], pred_bboxes.shape[1], 1),
                dtype=pred_bboxes.dtype,
                device=pred_bboxes.device,
            ),
            cls_score_list,
        ],
        dim=-1,
    )
    # print("output_merged", output_merged.shape)
    return non_max_suppression(
        output_merged,
        n_classes=n_classes,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        bbox_format="xyxy",
        max_det=max_det,
        predicts_objectness=False,
    )

def process_yolov6_outputs(outputs, conf_thresh=0.25):
    """
    Transforms YOLOv6 model outputs into bounding boxes in COCO format.

    Parameters
    ----------
    outputs : list 
        A list of a models outputs as numpy arrays, each of shape (1, 85, H, W)
    conf_thresh : float
        confidence threshold for filtering detections

    Returns:
    bboxes: list
        A list of bboxes
    """

    cls_score_list = []
    reg_distri_list = []

    for output in outputs:
        # Assuming output is of shape (1, 85, H, W)
        reg_distri = output[:, :4, :, :]  # First 4 channels are reg_distri
        out_cls = output[:, 4:, :, :]  # Remaining channels are cls_score

        reg_distri = torch.tensor(reg_distri)
        out_cls = torch.tensor(out_cls)
        # Append them to the lists
        reg_distri_list.append(reg_distri)
        cls_score_list.append(out_cls)
    
    cls_tensor = torch.cat(
        [cls_score_list[i].flatten(2) for i in range(len(cls_score_list))], dim=2
    ).permute(0, 2, 1)
    reg_tensor = torch.cat(
        [reg_distri_list[i].flatten(2) for i in range(len(reg_distri_list))], dim=2
    ).permute(0, 2, 1)
    
    features = [torch.randn(outputs[0].shape), torch.randn(outputs[1].shape), torch.randn(outputs[2].shape)]
        
    output = (features, cls_tensor, reg_tensor)
    boxes = process_to_bbox(output, conf_thres=conf_thresh)[0]

    return boxes

def process_segmentation_outputs(output_mask):
    """
    Transforms segmentation model outputs into segmentation masks.

    Parameters
    ----------
    output_mask : numpy array
        A numpy array of shape (1, C, H, W) where C is the number of classes

    Returns:
    mask: numpy array
        A segmentation mask
    """
    
    output_mask = output_mask[0]
    output_mask = output_mask.transpose(1, 2, 0)
    mask = np.argmax(output_mask, axis=-1)
    probs = np.max(output_mask, axis=-1)
    mask = mask.astype(np.uint8)
    return mask, probs
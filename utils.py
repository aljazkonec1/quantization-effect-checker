import numpy as np 
import cv2

def resize_and_pad(img: np.array, target_size = (512, 288)) -> np.array:
    img_h, img_w, _ = img.shape
    
    scale = target_size[0] / img_w
    new_w, new_h = target_size[0], min(int(img_h * scale), target_size[1])
    
    resized_image = cv2.resize(img, (new_w, new_h))
    pad_h = 0
    if new_h < target_size[1]:
        pad_h = target_size[1] - new_h
    
    top = pad_h // 2
    bottom = pad_h - top
    left = 0
    right = 0
    
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded_image


def update_menus(N1:int, N2: int, N3: int, N4: int, prediction_type: str = "detection") -> list:
    """Function to setup the button menus for the plotly figure.
    """
    buttons=[
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
        ]
    
    if prediction_type == "segmentation":
        buttons.append(dict(
            label="FPS vs F1 Score",
            method="update",
            args=[
                {"visible": [False] * (N1 + N2 + N3) + [True] * N4},
                {"title": "FPS vs F1 Score",
                    "xaxis": {"title": "FPS"},
                    "yaxis": {"title": "F1 Score"},
                    "barmode": "group",
                    "hovermode": "closest",
                    "showlegend": True}
            ]
        ))

    else: # detection
        # Button for FPS vs mAP@[IoU=0.50:0.95] Scatter Plot
        buttons.append(dict(
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
        ))
    
    return list(buttons)
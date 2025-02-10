import numpy as np 
import cv2
import os
import subprocess
import shutil
from rich.console import Console
import pandas as pd
import json
import time
console = Console()

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

def snpe_net_run_command(sftp, client, command, output_directory):
    console.print("[bold cyan]Running SNPE Net Run...[/bold cyan]")
    command = "cd /data/tests && rm -rf output/ && " + command + " && tar -czf output.tar.gz output"
    
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    error_output = stderr.read().decode().strip()
    if exit_status != 0:
        console.print(f"[red] Error running snpe-net-run. Got exit status: {exit_status} and exit code: {error_output} [/red]")
        raise Exception("Error running snpe-net-run")
    else:
        console.print("[green] snpe-net-run completed with exit status:[/green]", exit_status)

    sftp.get("/data/tests/output.tar.gz", "raw_outputs/output.tar.gz")
    
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    
    subprocess.run(["tar", "-xzf", "raw_outputs/output.tar.gz", "-C", "raw_outputs/"], check=True)
    shutil.move("raw_outputs/output", output_directory)
    # os.rename("raw_outputs/output", output_directory)
    os.remove("raw_outputs/output.tar.gz")

def send_files_to_device(sftp, client, host_base_path):
    """Copies files from host to device and puts them in the /data/tests/{last_dir_name} path.
    """
    base_name = os.path.basename(host_base_path)
    base_path = os.path.dirname(host_base_path)
    subprocess.run(["tar", "-czf", os.path.join(base_path, f"{base_name}.tar.gz"), "-C", base_path, base_name], check=True)
    archive_path = os.path.join(base_path, f"{base_name}.tar.gz")
    
    sftp.put(archive_path, f"/data/tests/{base_name}.tar.gz")
    stdin, stdout, stderr = client.exec_command(f"cd /data/tests && rm -rf {base_name} && tar -xzf {base_name}.tar.gz && rm {base_name}.tar.gz")
    exit_status = stdout.channel.recv_exit_status()
    error_output = stderr.read().decode().strip()

    if exit_status != 0:
        console.print(f"[red]Error copying files to device: {error_output}[/red]")
        raise Exception("Error copying files to device")

def dlc_graph_to_csv(csv_path: str):
    """
    Load the dlc graph as csv file and save the shape information as a json file.
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()
        
    start_row = next(i for i, line in enumerate(lines) if line.strip().startswith("Id,Name,Type"))

    end_row = next(i for i, line in enumerate(lines[start_row:], start=start_row) if line.strip().startswith("Note:"))

    df1 = pd.read_csv(csv_path, skiprows=start_row, nrows=end_row - start_row)
    df1 = df1.dropna()

    start_row = next(i for i, line in enumerate(lines) if line.strip().startswith("Output Name,Dimensions,Type,Encoding Info"))
    end_row = next(i for i, line in enumerate(lines[start_row:], start=start_row) if line.strip().startswith("Total parameters:"))
    
    df2 = pd.read_csv(csv_path, skiprows=start_row, nrows=end_row - start_row -1)

    # ony name and shape
    df1 = df1[["Name", "Out Dims"]]
    df1.columns = ["Name", "Shape"]
    df1["Shape"] = df1["Shape"].str.replace("x", ",")
    df2 = df2[["Output Name", "Dimensions"]]
    df2.columns = ["Name", "Shape"]
    df = pd.concat([df1, df2])
    df["Shape"] = df["Shape"].apply(lambda x: [int(d) for d in x.split(",")])
    df["Name"] = df["Name"].apply(lambda x: x.replace(".", "_").replace("/", "_").strip("_"))
    
    df.set_index("Name")["Shape"].to_json(csv_path.replace(".csv", ".json"), indent=4)

def flatten_dir(base_dir: str, dlc_outputs: str):
    dest_dir = base_dir + "/output"
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    
    config = {}
    dlc_output_dir = os.path.join(base_dir, dlc_outputs, "output/Result_0")
    for root, _, files in os.walk(dlc_output_dir):
        for file in files:
            # print(file)
            relative_path = os.path.relpath(root, dlc_output_dir)
            
            new_file_name = relative_path.replace(os.sep, "_") + f"_{file}"
            new_file_name = new_file_name.strip(".")
            new_file_name = new_file_name.strip("_")
            new_file_name = new_file_name.replace("_output_0", "")
            
            # Define the full paths
            source_path = os.path.join(root, file)
            destination_path = os.path.join(dest_dir ,new_file_name)
            ln = new_file_name.replace("..", ".").replace(".", "_")
            ln = ln.strip("_raw")
            config[ln] = "output/" + new_file_name
        
            # Move the file
            shutil.copy(source_path, destination_path)

    with open(os.path.join(base_dir, "config.json"), "w") as f:
        json.dump(config, f)
    
    shutil.rmtree(f"{base_dir}/{dlc_outputs}")

def generate_dlc_info_graph(model_path: str):
    model_dir = os.path.dirname(model_path)
    
    with open(f"{model_dir}/dlc-info-graph.txt", "w") as output_file:
        subprocess.run(["snpe-dlc-info", "-i", model_path, "-m", "-d"], stdout=output_file)
        subprocess.run(["snpe-dlc-info", "-i", model_path, "-m", "-d", "-s", f"{model_dir}/graph_config.csv" ], stdout=output_file)
        dlc_graph_to_csv(f"{model_dir}/graph_config.csv")
        os.remove(f"{model_dir}/graph_config.csv")
        
def run_throughput_test(sftp, client, base_dir, model_name):
    stdin, stdout, stderr = client.exec_command(f"cd /data/tests && snpe-throughput-net-run --duration 60 --use_dsp --container models/{model_name}.dlc --perf_profile balanced > throughputs/{model_name}.txt")
    exit_status = stdout.channel.recv_exit_status()
    sftp.get(f"/data/tests/throughputs/{model_name}.txt", f"{base_dir}/throughput.txt")
    time.sleep(2)

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
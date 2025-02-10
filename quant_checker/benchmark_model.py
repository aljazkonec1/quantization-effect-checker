import argparse
import os
import paramiko
from dotenv import load_dotenv 
import time
import paramiko.client
from util_functions import generate_dlc_info_graph, run_throughput_test
from make_dlc_info_json import generate_dlc_info_json
from process_onnx_model import run_onnx_model
from process_dlc_model import run_dlc_model
from per_layer_cycle_analysis import generate_per_layer_cycle_stats
from create_per_layer_statistics import compare_outputs
from rich.console import Console
from rich.panel import Panel
console = Console()

parser = argparse.ArgumentParser(description='Benchmark model')
parser.add_argument('--model_path', type=str, help='Path to dlc model')
parser.add_argument('--model_name', type=str, help='What name to save the model as')
parser.add_argument('--onnx_model_path', type=str, help='Path to the ONNX model')
parser.add_argument('--test_img_path', type=str, help='Relative path to the test image to be used for per layer analysis. The image will be converted to FP32 raw file and used on device')
parser.add_argument('--test_data_path', type=str, help='Path to the test data', default=None)
args = parser.parse_args()

load_dotenv()
hostname = os.getenv("HOSTNAME")
username = os.getenv("DEVICE_USER")
password = os.getenv("PASSWORD")

device_base_path = "/data/tests"
base_model_path = args.model_path
model_name = args.model_name
onnx_model_path = args.onnx_model_path
test_img_path = args.test_img_path
test_data_path = args.test_data_path


client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    client.connect(hostname, username=username, password=password)
    # console.print("[green]Connected to device successfully.[/green]")
except Exception as e:
    console.print("[bold red]Error connecting to device:[/bold red]", e)
    exit()
sftp = client.open_sftp()

console.print(Panel(f"[bold magenta]Testing Model: {model_name} [/bold magenta]"))

## run on onnx
console.print("[bold magenta]Processing ONNX file.[/bold magenta]")
onnx_model_dir = run_onnx_model(onnx_model_path, model_name, test_img_path)

# run on dlc
console.print(f"[bold magenta]Running DLC ... [/bold magenta]")
dlc_model_dir = run_dlc_model(sftp, client, base_model_path, model_name, "data_raw/" + model_name)

# generate dlc info graph
console.print("[bold magenta]Generating DLC info graph.[/bold magenta]")
model_path = dlc_model_dir + f"/{model_name}.dlc"
generate_dlc_info_graph(model_path)

# make per layer comparison csv
console.print("[bold magenta]Comparing outputs of onnx and dlc.[/bold magenta]")
compare_outputs(model_name, 
                onnx_config_path = onnx_model_dir + "/config.json", 
                dlc_config_path = dlc_model_dir + "/config.json",
                graph_info_path = dlc_model_dir + "/graph_config.json"
                )

# run throughput test
console.print("[bold magenta]Running throughput test.[/bold magenta]")
# run_throughput_test(sftp, client, dlc_model_dir, model_name)

raw_outputs_dir = "."
if test_data_path is not None:
    console.print("[bold magenta]Creating layer cycle stats.[/bold magenta]")
    raw_outputs_dir = generate_per_layer_cycle_stats(sftp, client, model_path, test_data_path)
    

console.print("[bold magenta]Generating info.json file.[/bold magenta]")
generate_dlc_info_json(model_name=model_name,
                        snpe_info_file_name="dlc-info-graph.txt",
                        onnx_model_layer_config=onnx_model_dir + "/config.json",
                        raw_outputs_dir=raw_outputs_dir
                        )
client.close()
time.sleep(2)

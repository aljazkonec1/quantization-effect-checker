import argparse
import os
import paramiko
from dotenv import load_dotenv 
import subprocess
import shutil
import time
import paramiko.client
from per_layer_profiler import per_layer_profiles
from make_dlc_info_json import generate_dlc_info_json
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
console = Console()

def create_model_dlc_dir(base_model_path, dlc_model_path, model_name):
    console.print(f"[blue]Creating model DLC directory in [bold yellow]{dlc_model_path}/{model_name}[/bold yellow][/blue]")
    os.makedirs(f"{dlc_model_path}/{model_name}", exist_ok=True)
    shutil.copyfile(base_model_path, f"{dlc_model_path}/{model_name}/{model_name}.dlc")
 
def snpe_net_run(sftp, client, input_bit_width, model_name, device_base_path, test_data_name):
    console.print("[bold green]Running SNPE Net Run...[/bold green]")
    bit_width_commands = {
        "int16": "--userbuffer_tfN 16 --userbuffer_tfN_output 16",
        "fp16": "--userbuffer_floatN 16 --userbuffer_floatN_output 16",
        "int8": "--userbuffer_tf8 --userbuffer_tf8_output"
    }
    commands = bit_width_commands.get(input_bit_width, "--userbuffer_tf8 --userbuffer_tf8_output")
    command = (
        f"cd {device_base_path} && rm -rf output/ && snpe-net-run --container models/{model_name}.dlc "
        f"--input_list {test_data_name}/inputs_raw.txt {commands} --use_dsp --use_native_input_files "
        "--use_native_output_files --perf_profile balanced && tar -czf output.tar.gz output"
    )
    stdin, stdout, stderr = client.exec_command(command)
    exit_status = stdout.channel.recv_exit_status()
    error_output = stderr.read().decode().strip()
    
    console.print("[bold blue]snpe-net-run completed with exit status:[/bold blue]", exit_status)
    
    sftp.get(f"{device_base_path}/output.tar.gz", f"raw_outputs/output.tar.gz")
    output_directory = f"raw_outputs/raw_output_{model_name}"
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    subprocess.run(["tar", "-xzf", "raw_outputs/output.tar.gz", "-C", "raw_outputs/"], check=True)
    os.rename("raw_outputs/output", output_directory)
    os.remove(f"raw_outputs/output.tar.gz")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark model')
    parser.add_argument('--model_path', type=str, help='Path to dlc model')
    parser.add_argument('--model_name', type=str, help='What name to save the model as')
    parser.add_argument('--test_data_path', type=str, help='Relative path to the test data to be used for benchmarking')
    parser.add_argument('--input_bit_width', type=str, choices=["int8", "int16", "fp16"], default="int8", help='Input bit width for the model')
    args = parser.parse_args()
    
    load_dotenv()
    hostname = os.getenv("HOSTNAME")
    username = os.getenv("DEVICE_USER")
    password = os.getenv("PASSWORD")
    
    dlc_model_path = "models_dlc"
    device_base_path = "/data/tests"
    base_model_path = args.model_path
    model_name = args.model_name
    test_data_path = args.test_data_path
    input_bit_width = args.input_bit_width
    
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname, username=username, password=password)
        # console.print("[green]Connected to device successfully.[/green]")
    except Exception as e:
        console.print("[bold red]Error connecting to device:[/bold red]", e)
        exit()
    
    console.print(Panel(f"[bold magenta]Testing Model: {model_name}[/bold magenta]"))
    
    sftp = client.open_sftp()
    test_data_name = test_data_path.split("/")[-1]
    stdin, stdout, stderr = client.exec_command(f"cd {device_base_path} && ls")
    output = stdout.read().decode()
    
    if test_data_name not in output:
        console.print(f"[bold yellow]Copying testing data \"{test_data_name}\" to device.[/bold yellow]")
        subprocess.run(["tar", "-czf", f"{test_data_path}.tar.gz", "-C", test_data_path.split("/")[0], test_data_name], check=True)
        sftp.put(f"{test_data_path}.tar.gz", f"{device_base_path}/{test_data_name}.tar.gz")
        stdin, stdout, stderr = client.exec_command(f"cd {device_base_path} && tar -xzf {test_data_name}.tar.gz && rm {test_data_name}.tar.gz")
        exit_status = stdout.channel.recv_exit_status()
    else:
        console.print(f"[bold green]Test data \"{test_data_name}\" already exists on device. No transfer needed.[/bold green]")
    
    create_model_dlc_dir(base_model_path, dlc_model_path, model_name)
    host_model_dir = f"{dlc_model_path}/{model_name}"
    host_model_path = f"{host_model_dir}/{model_name}.dlc"
    device_model_path = f"{device_base_path}/models/{model_name}.dlc"
    
    console.print(f"[blue]Copying model [bold yellow]\"{model_name}\"[/bold yellow] to device.[/blue]")
    sftp.put(host_model_path, device_model_path)
    
    console.print("[bold cyan]Generating DLC info graph.[/bold cyan]")
    with open(f"{host_model_dir}/dlc-info-graph.txt", "w") as output_file:
        subprocess.run(["snpe-dlc-info", "-i", host_model_path, "-m", "-d"], stdout=output_file)
    
    console.print("[bold cyan]Running throughput test.[/bold cyan]")
    stdin, stdout, stderr = client.exec_command(f"cd {device_base_path} && snpe-throughput-net-run --duration 60 --use_dsp --container models/{model_name}.dlc --perf_profile balanced > throughputs/{model_name}.txt")
    exit_status = stdout.channel.recv_exit_status()
    sftp.get(f"{device_base_path}/throughputs/{model_name}.txt", f"{host_model_dir}/throughput.txt")
    time.sleep(2)
    
    console.print("[bold cyan]Running detections on device.[/bold cyan]")
    snpe_net_run(sftp, client, input_bit_width, model_name, device_base_path, test_data_name)
    
    console.print("[bold cyan]Creating layer stats.[/bold cyan]")
    log_file = f"raw_outputs/raw_output_{model_name}/SNPEDiag.log"
    subprocess.run(["snpe-diagview", "--input_log", log_file, "--csv_format_version", "2", "--output", f"{host_model_dir}/layer_stats.csv"], 
                   check=True, 
                   stdout=subprocess.DEVNULL) 
    
    per_layer_profiles(f"{host_model_dir}/layer_stats.csv", f"{host_model_dir}/layer_stats.csv")
    
    console.print("[bold cyan]Generating info.json file.[/bold cyan]")
    generate_dlc_info_json(model_name=model_name,
                           base_dir=host_model_dir,
                           snpe_info_file_name="dlc-info-graph.txt",
                           text_file_path=f"{test_data_path}/inputs_raw.txt",
                           gt_labels_path="data/test/labels.json",
                           raw_outputs_dir=f"raw_outputs/raw_output_{model_name}",
                           )
    client.close()
    time.sleep(2)
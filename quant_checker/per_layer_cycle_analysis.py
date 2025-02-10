import os
import subprocess
from util_functions import send_files_to_device, snpe_net_run_command
from per_layer_profiler import per_layer_profiles
from rich.console import Console
console = Console()

def generate_per_layer_cycle_stats(sftp, client, model_path, test_data_path):
    model_name = os.path.basename(model_path)
    base_path = os.path.dirname(model_path)
    
    stdin, stdout, stderr = client.exec_command(f"cd /data/tests/&& ls")
    output = stdout.read().decode()
    output = output.split("\n")
    test_data_name = os.path.basename(test_data_path)
    
    if test_data_name not in output:
        console.print(f"[bold yellow]Copying testing data \"{test_data_name}\" to device.[/bold yellow]")
        
        send_files_to_device(sftp, client, test_data_path)
    else:
        console.print(f"[bold green]Test data \"{test_data_name}\" already exists on device. No transfer needed.[/bold green]")

    command = (
        f"snpe-net-run --container models/{model_name}.dlc "
        f"--input_list {test_data_name}/inputs_raw.txt --use_dsp --use_native_input_files --use_native_output_files --perf_profile balanced"
    )
    
    snpe_net_run_command(sftp, client, command, f"raw_outputs/{model_name}")
    
    log_file = f"raw_outputs/{model_name}/SNPEDiag.log"
    subprocess.run(["snpe-diagview", "--input_log", log_file, "--csv_format_version", "2", "--output", f"{base_path}/layer_stats.csv"], 
                check=True, 
                stdout=subprocess.DEVNULL) 
    
    per_layer_profiles(f"{base_path}/layer_stats.csv", f"{base_path}/layer_stats.csv")

    return f"{base_path}/layer_stats.csv"
    
    
    
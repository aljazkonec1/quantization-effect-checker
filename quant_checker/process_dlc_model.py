import os
import shutil
from util_functions import send_files_to_device, snpe_net_run_command, flatten_dir
from rich.console import Console

console = Console()
def run_dlc_model(sftp, client, dlc_model_path, model_name, test_img_raw_path):
    model_dir = "models_dlc/" + model_name
    os.makedirs(model_dir, exist_ok=True)
    host_model_path = f"{model_dir}/{model_name}.dlc"
    device_model_path = f"/data/tests/models/{model_name}.dlc"
    shutil.copyfile(dlc_model_path, host_model_path)
    
    console.print(f"[blue]Copying model [bold yellow]\"{model_name}\"[/bold yellow] to device.[/blue]")
    sftp.put(host_model_path, device_model_path)
    
    console.print("[bold cyan]Generating per layer outputs on SNPE...[/bold cyan]")
    send_files_to_device(sftp, client, test_img_raw_path)
    
    # execute on device
    command = f"snpe-net-run --container models/{model_name}.dlc --input_list {model_name}/inputs_raw.txt --debug --use_dsp --userbuffer_floatN_output 32 --perf_profile balanced --userbuffer_float"
    
    snpe_net_run_command(sftp, client, command, model_dir + "/raw_outputs")

    flatten_dir(model_dir, "raw_outputs")
    
    console.print("[bold green]DLC model run completed.[/bold green]")
    return model_dir
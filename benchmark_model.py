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


def create_model_dlc_dir(base_model_path, dlc_model_path, model_name):
    print(f"Creating model dlc directory in {dlc_model_path}/{model_name}")

    os.makedirs(f"{dlc_model_path}/{model_name}", exist_ok=True)
    shutil.copyfile(base_model_path, f"{dlc_model_path}/{model_name}/{model_name}.dlc")
    
def snpe_net_run(sftp, client, input_bit_width, model_name, device_base_path, test_data_name):
    if input_bit_width == "int16":
        bit_width_commands = "--userbuffer_tfN 16 --userbuffer_tfN_output 16"
    elif input_bit_width == "fp16":
        bit_width_commands = "--userbuffer_floatN 16 --userbuffer_floatN_output 16" 
    else: 
        bit_width_commands = "--userbuffer_tf8 --userbuffer_tf8_output"

    stdin, stdout, stderr = client.exec_command(f"cd {device_base_path} && rm -r output/ && snpe-net-run --container models/{model_name}.dlc --input_list {test_data_name}/inputs_raw.txt {bit_width_commands} --use_dsp --use_native_input_files --use_native_output_files --perf_profile balanced && tar -czf output.tar.gz output")
    exit_status = stdout.channel.recv_exit_status()
    
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
    parser.add_argument('--model_name', type=str, help='what name to save the model as')
    parser.add_argument('--test_data_path', type=str, help='relative path to the test data to be used for benchmarking')
    parser.add_argument('--input_bit_width', type=str, choices=["int8", "int16", "fp16" ], default="int8", help='input bit width for the model')
    args = parser.parse_args()
    load_dotenv()
    hostname = os.getenv("HOSTNAME") # IP
    username = os.getenv("DEVICE_USER") # root
    password = os.getenv("PASSWORD") # oelinux123
    
    dlc_model_path = "models_dlc"
    device_base_path = "/data/tests"
    base_model_path = args.model_path
    model_name = args.model_name
    test_data_path = args.test_data_path
    input_bit_width = args.input_bit_width
    
    client = paramiko.SSHClient()   
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    print(f"Testing model: {model_name}")
    try:
        client.connect(hostname, username=username, password=password)
        print("Connected to device")
    
    except Exception as e:
        print("Error connecting to device: ", e)
        exit()

    sftp = client.open_sftp()
    
    stdin, stdout, stderr = client.exec_command(f"cd {device_base_path} && ls")
    
    output = stdout.read().decode()
    test_data_name = test_data_path.split("/")[-1]
    
    if test_data_name not in output:
        print(f"Copying test data {test_data_name} to device.")
        
        subprocess.run(["tar", "-czf", f"{test_data_path}.tar.gz", "-C", test_data_path.split("/")[0],test_data_name ], check=True)
        sftp.put(f"{test_data_path}.tar.gz", f"{device_base_path}/{test_data_name}.tar.gz")
        stdin, stdout, stderr = client.exec_command(f"cd {device_base_path} && tar -xzf {test_data_name}.tar.gz && rm {test_data_name}.tar.gz")
        exit_status = stdout.channel.recv_exit_status()

    else:
        print(f"Test data {test_data_name} already exists on device. No transfer needed.")
    
    print(f"Creating dlc model directory.")
    create_model_dlc_dir(base_model_path, dlc_model_path, model_name)
    host_model_dir = f"{dlc_model_path}/{model_name}"
    host_model_path = f"{host_model_dir}/{model_name}.dlc"
    device_model_path = f"{device_base_path}/models/{model_name}.dlc"    
    
    print(f"Copying model to device.")
    sftp.put(host_model_path, device_model_path)
    
    print("Creating dlc info graph.")
    with open(f"{host_model_dir}/dlc-info-graph.txt", "w") as output_file:
        subprocess.run(["snpe-dlc-info", "-i", host_model_path, "-m", "-d"], stdout= output_file)
    
    
    print("Throughput test.")
    stdin, stdout, stderr = client.exec_command(f"cd {device_base_path} && snpe-throughput-net-run --duration 60 --use_dsp --container models/{model_name}.dlc --perf_profile balanced > throughputs/{model_name}.txt")
    exit_status = stdout.channel.recv_exit_status()
    sftp.get(f"{device_base_path}/throughputs/{model_name}.txt", f"{host_model_dir}/throughput.txt")
    time.sleep(2)
    
    print("Running detections on device.")
    snpe_net_run(sftp, client, input_bit_width, model_name, device_base_path, test_data_name)
    
    print("Creating layer stats.")    
    log_file = f"raw_outputs/raw_output_{model_name}/SNPEDiag.log"
    subprocess.run(["snpe-diagview", "--input_log", log_file, "--csv_format_version", "2", "--output", f"{host_model_dir}/layer_stats.csv"], check=True)
    per_layer_profiles(f"{host_model_dir}/layer_stats.csv", f"{host_model_dir}/layer_stats.csv")
    
    print("Creating info.json.")
    generate_dlc_info_json(model_name=model_name,
                           base_dir=host_model_dir,
                           snpe_info_file_name="dlc-info-graph.txt",
                           text_file_path=f"{test_data_path}/inputs_raw.txt",
                           gt_labels_path="data/test/labels.json",
                           raw_outputs_dir=f"raw_outputs/raw_output_{model_name}",
                           )
    
    client.close()
    time.sleep(2)
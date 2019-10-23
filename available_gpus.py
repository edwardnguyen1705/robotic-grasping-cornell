import subprocess

def get_available_gpus(mem_lim=1024):
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free', #memory.used
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print("GPUs memory available: {}".format(gpu_memory_map))
    
    gpus_available = [str(i) for i in range(len(gpu_memory_map)) if gpu_memory_map[i] > mem_lim]
    print("GPUs memory available > {} MB: {}".format(mem_lim, gpus_available))

    return gpus_available
    
if __name__ == '__main__':
    gpus_available = get_available_gpus(mem_lim=1024)   #
    


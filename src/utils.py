import torch


def get_available_device() -> tuple[torch.device, int]:
    """
    Returns the best available device for PyTorch computations.
    - If CUDA (GPU) is available, it returns 'cuda' and the number of GPUs.
    - If MPS (Apple GPU) is available, it returns 'mps' and 1 (for batch size handling).
    - Otherwise, it returns 'cpu' and 1.

    Returns:
        tuple[torch.device, int]: A tuple containing the device and the number of available devices.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        return device, gpu_count
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        return device, 1
    else:
        device = torch.device("cpu")
        return device, 1

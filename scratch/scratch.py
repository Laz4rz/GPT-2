import torch

def check_dtypes(device):
    dtypes = {
        'float32': torch.float32,
        'float64': torch.float64,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'uint8': torch.uint8,
        'int8': torch.int8,
        'int16': torch.int16,
        'short': torch.short,
        'int32': torch.int32,
        'int': torch.int,
        'int64': torch.int64,
        'long': torch.long,
    }.values()

    available_dtypes = []

    for dtype in dtypes:
        try:
            tensor = torch.tensor([1], dtype=dtype, device=device)
            available_dtypes.append(dtype)
        except Exception as e:
            print(f"Dtype {dtype} not supported on device {device}: {e}")

    return available_dtypes
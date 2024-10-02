import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())

if torch.backends.cudnn.is_available():
    cudnn_version = torch.backends.cudnn.version()
    print("cuDNN Version:", cudnn_version)
else:
    print("cuDNN is not available.")
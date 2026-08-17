import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version used by PyTorch:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(2000, 2000, device=device)
y = torch.randn(2000, 2000, device=device)

z = torch.matmul(x, y)

print("Device:", device)
print("Result shape:", z.shape)
print("GPU:", torch.cuda.get_device_name(0))
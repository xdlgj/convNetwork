import torch

flag = torch.cuda.is_available()
print(flag)
ngpu = 1

device = torch.device("cuda0" if flag and ngpu > 0 else "cpu")

cuda_version = torch.version.cuda
print(cuda_version)
cudnn_version = torch.backends.cudnn.version()
print(cudnn_version)
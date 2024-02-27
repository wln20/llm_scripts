"""
Description: Calculate the number of a model's parameters
"""
import torch

# model instantialization
model = XXX

# calculate all the parameters
def calc_all(model):
  total = sum(p.numel() for p in model.parameters())
  print("Total model params: %.2fM" % (total / 1e6))

# calculate the parameters of all the linear layers and conv1d linears
def calc_sep(model):
  total = 0
  total_linear = 0
  total_conv = 0
  for name, param in model.named_modules():
      # print(f'{name}\t{param.shape}')
      if isinstance(param, torch.nn.Linear):
          param_num = param.weight.numel()
          if param.bias is not None:
              param_num += param.bias.numel()
          total_linear += param_num
          total += param_num
      elif isinstance(param, torch.nn.Conv1d):
          param_num = param.weight.numel()
          if param.bias is not None:
              param_num += param.bias.numel()
          total_conv += param_num
          total += param_num
        
  print(f'linear_total: {total_linear / 1e6} MB')
  print(f'conv_total: {total_conv / 1e6} MB')
  print(f'total: {total / 1e6} MB')


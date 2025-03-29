import numpy as np

import torch


def get_device():
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda or use_mps else {}
    return device, kwargs

def predict(model, data_loader):
    device, kwargs = get_device()

    model.to(device)
    model.eval()

    all_inputs = []
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            outputs = model(inputs)

            all_inputs.append(inputs)
            all_outputs.append(outputs)
            all_targets.append(targets)

    all_inputs = torch.cat(all_inputs, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # the orign mse map to [1, 10]
    mse = torch.mean((9 * (all_outputs - all_targets)) ** 2)
    
    return all_inputs, all_outputs, all_targets, mse


import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def make_dataset(x_data, y_data, batch_size=256):
    tensor_x = torch.Tensor(x_data)
    tensor_y = torch.Tensor(y_data)

    return DataLoader(TensorDataset(tensor_x,tensor_y), batch_size=batch_size)

def train_one_epoch(epoch_index, model, training_loader, optimizer, loss_fn, device):
    running_loss = 0.
    last_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    model = model.to(device)
    for i, data in enumerate(training_loader):
        inputs, labels = data
        labels = labels.to(device).type(torch.LongTensor)
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        last_loss = running_loss # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        running_loss = 0.

    return last_loss


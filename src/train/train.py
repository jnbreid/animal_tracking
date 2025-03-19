from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch

"""
function to train a DistNet object for one epoch

Parameters:
- epoch_index (int)
- tb_writer (tensorboard writer object)
- model (DistNet object)
- loss_fkt (torch loss function)
- train_loader (torch data loader)
- test_loader (torch data loader)
- device (torch device)
- optimizer (torch optimizer)

Returns:
- loss (torch float)
"""
def train_one_epoch(epoch_index, tb_writer, model, loss_fkt, train_loader, test_loader, device, optimizer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        # get input and label onto device
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fkt(outputs, labels)
        loss.backward()

        # Adjust  weights
        optimizer.step()

        # add data in writer
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Trainingloss', last_loss, tb_x)
            running_loss = 0.

    return last_loss

"""
function to train a DistNet model for multiple epochs

Parameters:
- model (DistNet Object)
- loss_fkt (torch loss function)
- train_loader (torch data loader)
- val_loader (torch data loader)
- device (torch device)
- n_epochs (int)

Returns:
-
"""
def train(model, loss_fkt, train_loader, val_loader, device, n_epochs = 1500):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/nn_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = n_epochs
    best_vloss = 1_000_000.

    val_loss = []
    trn_loss = []

    for epoch in range(EPOCHS):
        if epoch%10 == 0:
            print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, model, loss_fkt, train_loader, val_loader, device)

        #print(avg_loss)

        running_vloss = 0.0
        model.eval()

        # Disable gradient computation and reduce memory consumption.

        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fkt(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        val_loss.append(avg_vloss)

        trn_loss.append(avg_loss)

        writer.add_scalars('Training vs. Validation Loss', { 'Training' : avg_loss, 'Validation' : avg_vloss }, epoch_number + 1)
        writer.flush()


        model_path = '/content/models/model_{}'.format(epoch_number)
        torch.save(model.state_dict(), model_path)

        epoch_number += 1
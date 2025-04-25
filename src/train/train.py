from datetime import datetime

import torch

def train_one_epoch(epoch_index, model, loss_fkt, train_loader, test_loader, device, optimizer, tb_writer = None):
    """
    Trains a DistNet model for a single epoch.

    Iterates over the training data, performs forward and backward passes, and updates model weights.

    Args:
        epoch_index (int): Current epoch number.
        tb_writer (SummaryWriter): TensorBoard writer object for logging. (install tensorflow to be able to use tensoboard)
        model (torch.nn.Module): The DistNet model being trained.
        loss_fkt (torch.nn.Module): Loss function used to compute loss.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset (unused here).
        device (torch.device): The device to run the training on.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.

    Returns:
        float: The last computed loss value (averaged over the most recent 10 batches).
    """
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
            if tb_writer is not None:
                tb_writer.add_scalar('Trainingloss', last_loss, tb_x)
            running_loss = 0.

    return last_loss

#from torch.utils.tensorboard import SummaryWriter

def train(model, loss_fkt, train_loader, val_loader, device, n_epochs = 1500):
    """
    Trains a DistNet model over multiple epochs.

    Performs training and validation loops, saves the model at each epoch, and logs both training
    and validation losses. Uses `train_one_epoch` for training per epoch.

    Args:
        model (torch.nn.Module): The DistNet model to be trained.
        loss_fkt (torch.nn.Module): The loss function used to optimize the model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): The device to run the training on.
        n_epochs (int, optional): Number of training epochs. Defaults to 1500.

    Returns:
        None
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = None #SummaryWriter('runs/nn_{}'.format(timestamp)) #(install tensorboard to use)
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
        avg_loss = train_one_epoch(epoch_number, model, loss_fkt, train_loader, val_loader, device, tb_writer = writer)

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
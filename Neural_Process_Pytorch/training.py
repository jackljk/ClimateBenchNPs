import torch
from random import randint
from Neural_Process_Pytorch.neural_process import NeuralProcessImg
from torch import nn
from torch.distributions.kl import kl_divergence
from Neural_Process_Pytorch.utils import (context_target_split, batch_context_target_mask,
                   img_mask_to_np_input)


class NeuralProcessTrainer():
    """
    Class to handle training of Neural Processes for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq
        
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, epochs):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*self.num_context_range)
                num_extra_target = randint(*self.num_extra_target_range)

                # Create context and target points and apply neural process
                # TODO : Fix data for actual data shape
                X, y = data  # data is a tuple (img, label)
                batch_size = X.size(0)
                
                # TODO : Implement the function to create context and target mask for the grid
                context_mask, target_mask = \
                    _create_context_target_mask(
                        X.size()[1:], num_context, num_extra_target, batch_size
                    )

                X = X.to(self.device)
                y = y.to(self.device)
                context_mask = context_mask.to(self.device)
                target_mask = target_mask.to(self.device)

                p_y_pred, q_target, q_context = \
                    self.neural_process(img, context_mask, target_mask) # TODO: Fix this line

                # Calculate y_target as this will be required for loss
                # TODO : Implement the function get back the y_target from the mask
                _, y_target = _get_sparse_input(
                    X, y, target_mask
                )

                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                self.steps += 1

                if self.steps % self.print_freq == 0:
                    print("iteration {}, loss {:.3f}".format(self.steps, loss.item()))

            print("Epoch: {}, Avg_loss: {}".format(epoch, epoch_loss / len(data_loader)))
            self.epoch_loss_history.append(epoch_loss / len(data_loader))

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl


def _get_sparse_input(X, y, mask):
    """
    Get sparse input data for training given the inputs and the mask
    """
    # Find the indices where the mask is 1
    mask_indices = mask.nonzero(as_tuple=True)
    
    # Gather the corresponding elements from X and y
    X_sparse = torch.cat([
        X[mask_indices[0], 0, mask_indices[1], mask_indices[2]].unsqueeze(1),
        X[mask_indices[0], 1, mask_indices[1], mask_indices[2]].unsqueeze(1),
        X[mask_indices[0], 2, mask_indices[1], mask_indices[2]].unsqueeze(1),
        X[mask_indices[0], 3, mask_indices[1], mask_indices[2]].unsqueeze(1),
        mask_indices[1].unsqueeze(1).float(),  # Add latitude indices
        mask_indices[2].unsqueeze(1).float()   # Add longitude indices
    ], dim=1)
    
    y_sparse = torch.cat([
        y[mask_indices[0], 0, mask_indices[1], mask_indices[2]].unsqueeze(1),
        mask_indices[1].unsqueeze(1).float(),  # Add latitude indices
        mask_indices[2].unsqueeze(1).float()   # Add longitude indices
    ], dim=1)
    
    return X_sparse, y_sparse

def _get_random_lat_lng_mask(context_point_shape, num_context, num_extra_target):
    aerosol_dim, lat_dim, lng_dim = context_point_shape
    
    # Empty mask
    context_mask = torch.zeros((lat_dim, lng_dim))
    target_mask = torch.zeros((lat_dim, lng_dim))

    # random lat and lng
    context_lat = np.random.randint(0, lat_dim, num_context)
    context_lng = np.random.randint(0, lng_dim, num_context)
    target_lat = np.random.randint(0, lat_dim, num_context + num_extra_target)
    target_lng = np.random.randint(0, lng_dim, num_context + num_extra_target)
    # set mask to 1
    context_mask[context_lat, context_lng] = 1
    target_mask[target_lat, target_lng] = 1
    
    return context_mask, target_mask

def _create_context_target_mask(data_size, num_context, num_extra_target, batch_size, one_mask=True):
    aerosol_dim, lat, lng = data_size
    batch_context_mask = torch.zeros(batch_size, lat, lng)
    batch_target_mask = torch.zeros(batch_size, lat, lng)
    
    if one_mask:
        context_mask, target_mask = _get_random_lat_lng_mask((1, lat, lng), num_context, num_extra_target)
        for i in range(batch_size):
            batch_context_mask[i] = context_mask
            batch_target_mask[i] = target_mask
        
        return batch_context_mask, batch_target_mask
    else:
        for i in range(batch_size):
            batch_context_mask[i], batch_target_mask[i] = _get_random_lat_lng_mask((1, lat, lng), num_context, num_extra_target)
        
    return batch_context_mask, batch_target_mask
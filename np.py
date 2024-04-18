from model import *

"""Neural Processes modules --> from scratch attempt"""
class Encoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim


        self.conv2d_td = TimeDistributed(nn.Conv2d(4, 20, kernel_size=(3, 3), padding='same'), batch_first=True)
        self.avg_pool_td = TimeDistributed(nn.AvgPool2d(2), batch_first=True)
        self.global_avg_pool_td = TimeDistributed(nn.AdaptiveAvgPool2d(1), batch_first=True)
        self.lstms = nn.LSTM(25, 25, batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.hidden = nn.Linear(1*96*144, r_dim)
        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        # x has shape: (batch_size, num_years, height, width, num_channels)
        # y has shape: (batch_size, 1, height, width)
        
        # Process spatial features with convolutional and pooling layers
        x = self.conv2d_td(x)
        x = self.avg_pool_td(x)
        x = self.global_avg_pool_td(x)
        x = x.view(x.size(0), x.size(1), -1)  # Reshape for LSTM: Flatten spatial dimensions

        # Combine x and y before LSTM if needed, or process separately based on your architecture's needs
        combined = torch.cat((x, y), dim=-1)  # Assuming y is already processed to match x's processed shape
        
        # Temporal processing with LSTM
        lstm_out, (hidden, _) = self.lstms(combined)
        # You can use hidden, or further process lstm_out if needed

        # Generate r_i from the final LSTM output or hidden state
        r_i = self.hidden(hidden[-1])  # Using the last layer's hidden state

        return r_i

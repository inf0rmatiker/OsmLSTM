import load_data
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.autograd import Variable



def print_tensor(tensor, start, end):
        print('[', end="", flush=True)
        for i in range(start, end):
            print('%.8f ' % tensor[i].item(), end="", flush=True)
        print(']')

'''
Note: tutorial from https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
used to create LSTM class.

'''
class LSTM(nn.Module):

    ''' input_size
            Size of input tensor (number of features). For zoom level 1, this is 4.

        window_size
            How many days are input.

        hidden_size
            How many neurons are in each hidden layers.

        output_size
            How many neurons in the output (how many predictions). For zoom level 1, this is 4.

    '''
    def __init__(self, input_size, window_size=1, hidden_size=4, output_size=4):
        super(LSTM, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # LSTM algorithm accepts 3 inputs: previous hidden state, previous cell state, and current input.
        self.lstm   = nn.LSTM(input_size, hidden_size) # LSTM layer: input -> hidden_layer
        self.linear = nn.Linear(hidden_size, output_size) # Linear layer: hidden_layer -> output

        # Contains the previous hidden and cell state.
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_size),
                            torch.zeros(1, 1, self.hidden_size))

    def forward(self, input_seq):
        # Pass input_seq into LSTM layer, which outputs hidden and cell states at current time step.
        lstm_output, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_output.view(len(input_seq), 1, -1))
        return predictions[-1]


def main():

    X_WIDTH  = 165
    Y_HEIGHT = 87

    # Variables for LSTM
    WINDOW_SIZE = 10
    INPUT_SIZE  = X_WIDTH * Y_HEIGHT
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = X_WIDTH * Y_HEIGHT
    EPOCHS = 25

    # Variables for data loader
    DIRECTORY = './zoom_10_subset/'

    max_count = load_data.find_max_count(DIRECTORY)

    train_f, train_t, valid_f, valid_t, test_f, test_t = load_data.get_data_sets(DIRECTORY, WINDOW_SIZE, max_count)

    train_set    = torch.utils.data.TensorDataset(train_f, train_t)
    valid_set    = torch.utils.data.TensorDataset(valid_f, valid_t)
    test_set     = torch.utils.data.TensorDataset(test_f, test_t)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=1, shuffle=False)

    '''
    model = LSTM(INPUT_SIZE, WINDOW_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    iteration = 0
    iteration_break = -1
    interval_print  = -1


    for epoch in range(EPOCHS):
        for seq, labels in train_loader:
            seq = seq.view(-1, Y_HEIGHT*X_WIDTH)
            labels = labels.view(-1, Y_HEIGHT*X_WIDTH)

            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_size),
                                 torch.zeros(1, 1, model.hidden_size))

            y_prediction = model(seq)

            single_loss = loss_func(y_prediction, labels) # calculate loss for single prediction
            single_loss.backward() # propagate loss backward
            optimizer.step()

            seq = seq * max_count
            labels = labels * max_count
            y_prediction = y_prediction * max_count

            if (iteration % interval_print == 0):
                print('Iteration %d' % iteration)
                print('Sequence:   %s\nLabels:     %s\nPrediction: %s' % (str(seq), str(labels), str(y_prediction)) )
                print('Loss (MSE): %s\n' % single_loss.item())
                #print_tensor(labels[0], 4000, 4050)
                #print_tensor(y_prediction, 30)

            if (iteration == iteration_break):
                break

            iteration += 1

        if (epoch % 5 == 0):
            print('Epoch: %d: loss: %10.8f' % (epoch, single_loss.item()))

    model.eval()
    for seq, labels in valid_loader:
        with torch.no_grad():
            seq = seq.view(-1, Y_HEIGHT*X_WIDTH)
            labels = labels.view(-1, Y_HEIGHT*X_WIDTH)

            model.hidden_cell = (torch.zeros(1, 1, model.hidden_size),
                                 torch.zeros(1, 1, model.hidden_size))

            y_prediction = model(seq)
            single_loss = loss_func(y_prediction, labels)

            seq = seq * max_count
            labels = labels * max_count
            y_prediction = y_prediction * max_count


            print('TEST Iteration %d' % iteration)
            print('TEST Sequence:   %s\nLabels:     %s\nPrediction: %s' % (str(seq), str(labels), str(y_prediction)) )
            print('TEST Loss (MSE): %s\n' % single_loss.item())
            print(int(y_prediction[0][0]*max_count))
            print(int(labels[0][0]*max_count))
            break
    '''

if __name__ == '__main__':
    main()

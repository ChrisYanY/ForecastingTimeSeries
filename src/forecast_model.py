
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # input_seq: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(input_seq)
        # We only care about the last time step output for prediction
        last_time_step = lstm_out[:, -1, :] 
        predictions = self.linear(last_time_step)
        return predictions

def train_model(model, X_train, y_train, epochs=25, lr=0.001):
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)

    model.train()
    for i in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train_t)
        single_loss = loss_function(y_pred, y_train_t)
        single_loss.backward()
        optimizer.step()
        
        if (i+1) % 5 == 0:
            print(f'epoch: {i+1:3} loss: {single_loss.item():10.8f}')

    return model

def predict(model, X_input):
    model.eval()
    with torch.no_grad():
        X_input_t = torch.FloatTensor(X_input)
        preds = model(X_input_t)
        return preds.numpy()

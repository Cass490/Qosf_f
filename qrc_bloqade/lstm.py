import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1, num_layers=1, dropout=0.1):
        super(LstmModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer= num_layers
        self.lstm= nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Get the last time step's output
        return out

class Sequential_qrc:
    def __init__(self, input_size, hidden_size=64, output_size=1, num_layers=1, dropout=0.1 ,seq_length= 3, batch_size=32, learning_rate= 0.001, num_epochs=100):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = LstmModel(input_size, hidden_size, output_size, num_layers, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def prep_sequence(self, features):
        sequences = []
        labels=[]
        for i in range(len(features) - self.seq_length ):
            seq = features[i:i + self.seq_length]
            target= features[i + self.seq_length]
            sequences.append(seq)
            labels.append(target)
        return np.array(sequences), np.array(labels)
    
    def fit(self, features, labels):
        if len(features)<= self.seq_length:
             print(f"Warning: Not enough samples ({len(features)}) for sequence length {self.seq_length}")
            # Reduce sequence length to accommodate smaller dataset
             self.seq_length = max(1, len(features) // 2)
             print(f"Reducing sequence length to {self.seq_length}")
        
        X = torch.tensor(features, dtype=torch.float32).to(self.device)
        y = torch.tensor(labels, dtype=torch.float32).to(self.device)

        dataset =TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
         # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Calculate loss
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss/len(dataloader):.6f}')

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            # Convert to PyTorch tensor
            X = torch.tensor(x, dtype=torch.float32).to(self.device)
            
            # Generate predictions
            predictions = self.model(X).cpu().numpy()
            
        return predictions
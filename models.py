import torch
import torch.nn as nn
import numpy as np
from multilabel_metrics import accuracy_multilabel

# converts [batch, channel, w, h] to [batch, units]
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Print(nn.Module):
    def forward(self, x):
        print(f'{x.size()}')
        return x


class ECGNet(nn.Module):
    def __init__(self, print_layer=False):
        super(ECGNet, self).__init__()

        self.final = nn.Sequential(
            # input: [n, 12, 1000]
            nn.Conv1d(in_channels=12, out_channels=24, kernel_size=17, stride=2), # output: [n, 24, 492]
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=10, stride=2),  # output: [n, 48, 242]
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=48, out_channels=96, kernel_size=10, stride=2),  # output: [n, 96, 117]
            nn.BatchNorm1d(96),
            nn.MaxPool1d(3, stride=2), # output: [n, 96, 58]
            nn.ReLU(),
            nn.Dropout(0.5),
            Flatten(),
            nn.Linear(5568, 100),
            nn.ReLU(),
            nn.Linear(100, 25),
            nn.ReLU(),
            nn.Linear(25, 5),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        output = self.final(X)
        return output


class ECGWithMetaNet(nn.Module):
    def __init__(self, print_layer=False):
        super(ECGWithMetaNet, self).__init__()

        self.signal_layers = nn.Sequential(
            # input: [n, 12, 1000]
            nn.Conv1d(in_channels=12, out_channels=24, kernel_size=17, stride=2), # output: [n, 24, 492]
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=10, stride=2),  # output: [n, 48, 242]
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=48, out_channels=96, kernel_size=10, stride=2),  # output: [n, 96, 117]
            nn.BatchNorm1d(96),
            nn.MaxPool1d(3, stride=2), # output: [n, 96, 58]
            nn.ReLU(),
            nn.Dropout(0.5),
            Flatten(),
            nn.Linear(5568, 100),
            nn.ReLU(),
            nn.Linear(100, 25),
            nn.ReLU()
        )   
        
        self.meta_layers = nn.Sequential(
           nn.Linear(48, 24),
           nn.ReLU(),
           nn.Dropout(0.5),
           nn.Linear(24, 10),
           nn.ReLU()
        )
        
        self.final = nn.Sequential(
            nn.Linear(25 + 10, 5),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        X, Meta = inputs
        signal_features = self.signal_layers(X)
        meta_features = self.meta_layers(Meta)
        features = torch.cat((signal_features, meta_features), dim=1)
        output = self.final(features)
        return output
    

# To train models independently from meta-features existence 
class ModelWrapper:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.criterion = torch.nn.BCELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def iter_batch(self, X, X_meta, y, batch_size=128):
        for i in range(0, X.shape[0], batch_size):
            yield (X[i:i + batch_size],
                   None if X_meta is None else X_meta[i:i + batch_size],
                   y[i:i + batch_size])
            
    def train_n_epochs(self, X_train, X_train_meta, y_train, X_val, X_val_meta, y_val,
                       path, epochs=75, verbose=True, batch_size=128):
        X_train_T = torch.Tensor(np.transpose(X_train, [0, 2, 1]))
        X_train_meta_T = None if X_train_meta is None else torch.Tensor(X_train_meta)

        X_val_T = torch.Tensor(np.transpose(X_val, [0, 2, 1]))
        X_val_meta_T = None if X_val_meta is None else torch.Tensor(X_val_meta)
        
        # Calculate accuracy
        train_acc = []
        val_acc = []
        best_val_acc = 0.
        
        for epoch in range(0, epochs):
            # Iterates through the train folds
            for (X_train_b, X_train_meta_b, y_train_b) in self.iter_batch(X_train, X_train_meta, y_train, batch_size):
                X_train_b_T = torch.Tensor(np.transpose(X_train_b, [0, 2, 1]))
                X_train_meta_b_T = None if X_train_meta_b is None else torch.Tensor(X_train_meta_b)
                y_train_b_T = torch.Tensor(y_train_b)
                self.fit((X_train_b_T, X_train_meta_b_T), y_train_b_T, verbose=verbose)

            # Train score
            y_train_pred = self.predict((X_train_T, X_train_meta_T)).detach().numpy()
            train_acc.append(accuracy_multilabel(y_train, y_train_pred, total_only=True))
            
            # Validation score
            y_val_pred = self.predict((X_val_T, X_val_meta_T)).detach().numpy()
            val_acc.append(accuracy_multilabel(y_val, y_val_pred, total_only=True))

            if val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]
        
            print(f'EPOCH {epoch}')
            print(f'Train accuracy: {train_acc[-1]}')
            print(f'Validation accuracy: {val_acc[-1]}')
        
        # Save the model
        self.save(path)

        return train_acc, val_acc, best_val_acc

    def fit(self, X, y, verbose=False):
        # Supports meta features
        if X[1] is None:
            X = X[0]
        self.model.train()
        self.opt.zero_grad()
        prediction = self.model(X)
        loss = self.criterion(prediction, y)
        if verbose:
            print(loss)
        loss.backward()
        self.opt.step()
                              
    def predict(self, X):
        if X[1] is None:
            X = X[0]
        self.model.eval()
        return self.model(X)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
    
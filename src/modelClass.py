# -*- coding: utf-8 -*-
"""
Created on Feb 2025

@author: Albert Alcalde
"""
import os
from datetime import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelClass():
    def __init__(self, name='model'):

        self.result_dir = os.path.join('Results', name)
        os.makedirs(self.result_dir, exist_ok=True)

    # %% Setting the Training and Validation Datasets
    def set_dynamics_dataset(self, train_dataset, val_dataset=None):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val = val_dataset is not None

    # %% Setting the main model, its optimizer and learning curves
    def set_model(self, model, multistep = True, weight_decay=0.01, opt_kwargs={}, device=Device):
        self.model = model.to(device)
        self.multistep = multistep

        # Use AdamW optimizer with specified weight_decay
        self.optimizer = optim.AdamW(self.model.parameters(), weight_decay=weight_decay, **opt_kwargs)
        self.train_loss = []
        self.val_loss = []

    # %% Saving and Loading the state_dict
    def save_state_dict(self, path=None):
        date_str = datetime.now().strftime("%d-%m_%H:%M")  # Includes hour and minute
        if path is None:
            path = self.result_dir + '/state_dict_' + date_str + '.pth'

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
        }, path)

    def load_state_dict(self, model_path=None):
        if model_path is None:
            path = os.path.join(self.result_dir, "state_dict.pth")
        else:
            path = model_path #os.path.join("Results", model_path, "state_dict.pth")

        if not os.path.exists(path):
            raise FileNotFoundError(f"State dict file not found: {path}")

        checkpoint = torch.load(path, map_location=torch.device("cpu"))

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load losses if they exist
        self.train_loss = checkpoint.get('train_loss', [])
        self.val_loss = checkpoint.get('val_loss', [])

    # %% The main model
    def train(self, criterion=nn.MSELoss(), epochs=50, batch_size=64, device=Device):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        if self.val:
            val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs * len(train_loader))
        self.model.to(device)

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            # Training:
            losses = []
            self.model.requires_grad_().train()
            for x, y in train_loader:
                # x: [B, n, d], y: [B, m, d]
                x, y = x.to(device), y.to(device)

                self.optimizer.zero_grad()
                # y_pred = self.model(x)

                if self.multistep:
                    # Autoregressive multistep prediction with sliding window
                    m = y.shape[1]
                    x_roll = x                                      # [B, n, d]
                    y_preds = []
                    for _ in range(m):
                        y_t = self.model(x_roll)      # expected [B, 1, d]
                        # If model returns [B, d], unsqueeze to [B, 1, d]
                        if y_t.dim() == 2:
                            y_t = y_t.unsqueeze(1)
                        y_preds.append(y_t)

                        # shift window forward in time: drop oldest, append prediction
                        x_roll = torch.cat([x_roll[:, 1:, :], y_t], dim=1)  # still [B, n, d]
                    
                    y_pred = torch.cat(y_preds, dim=1)   # [B, m, d]
                    del x_roll, y_preds
                else:
                    # plain single-step prediction
                    y_pred = self.model(x)
                
                loss = criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                scheduler.step()
                losses.append(loss.item())
                del x, y, y_pred
            train_loss = np.mean(losses)
            self.train_loss.append(train_loss)

            # Validation:
            if self.val:
                self.model.requires_grad_(False).eval()
                losses = []
                for x, y in val_loader:
                    x, y  = x.to(device), y.to(device)
                    y_pred = self.model(x)
                    loss = criterion(y_pred, y)
                    losses.append(loss.item())
                    del x, y, y_pred
                val_loss = np.mean(losses)
                self.val_loss.append(val_loss)

            # Printing the results:
            print(f'Train Loss: {train_loss:.6f}')
            if self.val:
                print(f'  Val Loss: {val_loss:.6f}')
            print('~'*60)
            if train_loss == min(self.train_loss):
                self.save_state_dict()

    def AR_sliding(self, x0_td, vmin, vmax, steps, mean=None, std=None):
        """
        Autoregressive sliding prediction for normalized or standardized data.

        Args:
            x0_td (np.ndarray): Initial sequence, shape [1, n, d] or [n, d].
            vmin (np.ndarray): Minimum values for normalization (ignored if mean/std provided).
            vmax (np.ndarray): Maximum values for normalization (ignored if mean/std provided).
            steps (int): Number of prediction steps.
            mean (np.ndarray, optional): Mean for standardization.
            std (np.ndarray, optional): Std for standardization.

        Returns:
            np.ndarray: Predicted sequence, shape [d, steps].
        """
        self.model.eval()
        d, n = x0_td.shape
        preds = np.zeros((steps, d))

        # Normalize or Standardize
        if mean is not None and std is not None:
            x0_td = (x0_td - mean) / std
            unnormalize = lambda x: x * std + mean
        else:
            x0_td = 2 * (x0_td - 0.5 * (vmax + vmin)) / (vmax - vmin)
            unnormalize = lambda x: x * 0.5 * (vmax - vmin) + 0.5 * (vmax + vmin)

        preds[:n] = x0_td.squeeze(0).T  # add initial sequence as ground truth to the 'prediction'

        sequence = torch.tensor(x0_td.squeeze(0).T, dtype=torch.float32).clone().to(Device)
        sequence = sequence.unsqueeze(0)

        for t in range(n, steps):
            with torch.no_grad():
                pred_seq = self.model(sequence)  # HSA: [1, d] or CSA: [1, n, d]

                if pred_seq.dim() == 2:
                    pred_seq = pred_seq.unsqueeze(1)  # [1,1,d]
                else:
                    pred_seq = pred_seq[:, -1:, :]  # Take last token, [1,1,d]

            preds[t] = pred_seq.cpu().numpy()
            sequence = torch.cat([sequence[:, 1:, :], pred_seq], dim=1)  # Shift sequence forward

        # Unnormalize or Unstandardize
        preds = unnormalize(preds.T)
        return preds
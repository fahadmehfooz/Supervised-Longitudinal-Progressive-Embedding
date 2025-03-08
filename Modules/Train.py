import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' 


import random
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchviz import make_dot

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import umap
from sklearn.linear_model import LogisticRegression
import pickle


warnings.filterwarnings("ignore")


def set_global_seed(seed):
    # Set all Python-related seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure deterministic CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  


class AE(nn.Module):
    def __init__(self, dims, activation="tanh", dropout=0.0, batch_norm=False):
        super(AE, self).__init__()
        self.dims = dims  # Store dims for later use
        self.encoder = self.build_layers(dims[:dims.index(min(dims)) + 1], activation, dropout, batch_norm)
        self.decoder = self.build_layers(dims[dims.index(min(dims)):], activation, dropout, batch_norm, reverse=True)

    def build_layers(self, layer_dims, activation, dropout, batch_norm, reverse=False):
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if not reverse or i < len(layer_dims) - 2:
                if isinstance(activation, str):
                    if activation == "sigmoid":
                        layers.append(nn.Sigmoid())
                    elif activation == "leakyrelu":
                        layers.append(nn.LeakyReLU())
                    elif activation == "tanh":
                        layers.append(nn.Tanh())
                    else:
                        layers.append(nn.ReLU())
                else:
                    layers.append(activation())  # Instantiate activation function if class is passed
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view((-1, self.encoder[0].in_features))
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        x = x.view((-1, self.encoder[0].in_features))
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def get_encoded_dim(self):
        return min(self.dims) 


def compute_directional_loss_tensor(encoded_data, train_data_identifiers):

    sorted_indices = train_data_identifiers.sort_values(["RID", 'EXAMDATE']).index
    sorted_RIDs = train_data_identifiers.loc[sorted_indices, "RID"].values
    sorted_encoded_data = encoded_data[sorted_indices]



    directional_losses = []

    for i in range(sorted_encoded_data.size(0) - 2):
        # Has to be consecutive RIDs for z1, z2, z3
        if sorted_RIDs[i] == sorted_RIDs[i+1] and sorted_RIDs[i+1] == sorted_RIDs[i+2]:
            z1 = sorted_encoded_data[i]
            z2 = sorted_encoded_data[i+1]
            z3 = sorted_encoded_data[i+2]

            # Computing vector1 and vector 2
            vector_12 = z2 - z1
            vector_23 = z3 - z2

            # Normalizing vectors to avoid division by zero
            vector_12_norm = torch.norm(vector_12) + 1e-12
            vector_23_norm = torch.norm(vector_23) + 1e-12

            # Calculating cosine similarity between two vectors, this would give direction
            cos = torch.sum(vector_12 * vector_23) / (vector_12_norm * vector_23_norm)

            loss = (1.0 - cos)
            
            directional_losses.append(loss)

    # Calculating the average directional loss
    if directional_losses:
        avg_directional_loss = torch.stack(directional_losses).mean()
    else:
        avg_directional_loss = torch.tensor(0.0, requires_grad=True)

    return avg_directional_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def initialize_weights(model, seed=42):
    torch.manual_seed(seed)
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)



def train_AE(model, train_data, test_data,  X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,  train_data_identifiers, test_data_identifiers, criterion, optimizer_class, lr,  epochs=500, lambda_reconstruction = 1.0,
                       lambda_directional=1.0, directional_loss_enabled = True, continuous_df = None, return_losses=False ):

    torch.use_deterministic_algorithms(True)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)  
    set_global_seed(42)

    print(f"Training for {epochs} epochs with optimizer: {str(optimizer_class)}, learning rate: {lr}")


    print()
    # Initializing model and set deterministic algorithms
    initialize_weights(model, seed=42)
    torch.use_deterministic_algorithms(True)

    optimizer = optimizer_class(list(model.parameters()), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    loss_history = []

    # Training loop
    for epoch in range(epochs):
        set_seed(42)

        optimizer.zero_grad()


        # Forward pass
        encoded = model.encode(X_train_tensor)
        reconstructed = model.decode(encoded)



        # Calculating reconstruction loss
        recon_loss = criterion(reconstructed, X_train_tensor)


        directional_loss = 0.0

        if directional_loss_enabled:

          directional_loss = compute_directional_loss_tensor(encoded, train_data_identifiers)

        
        # Total loss
        total_loss = lambda_reconstruction * recon_loss  + lambda_directional * directional_loss

        # Backward pass and optimization step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)


        if epoch % 50 == 0:
          loss_entry = {
            'epoch': epoch,
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item() 
          }


        if directional_loss_enabled:
            loss_entry['directional_loss'] = lambda_directional* directional_loss.item()

        if epoch % 50 == 0:
          loss_history.append(loss_entry)

    train_loss_df = pd.DataFrame(loss_history)
    display(train_loss_df)

    model.eval()
    with torch.no_grad():

        encoded_test = model.encode(X_test_tensor)
        reconstructed_test = model.decode(encoded_test)
        
        test_recon_loss = criterion(reconstructed_test, X_test_tensor)
        directional_loss_test = torch.tensor(0.0)
        
        if directional_loss_enabled:
            directional_loss_test = compute_directional_loss_tensor(encoded_test, test_data_identifiers)

        total_loss_test = (lambda_reconstruction * test_recon_loss) + (lambda_directional * directional_loss_test)

    final_test_losses = {
        "Metric": ["Reconstruction Loss", "Directional Loss", "Total Loss"],
        "Train": [
            train_loss_df["recon_loss"].iloc[-1],
            train_loss_df["directional_loss"].iloc[-1] if directional_loss_enabled else 0.0,
            train_loss_df["total_loss"].iloc[-1]
        ],
        "Test": [
            test_recon_loss.item(),
            directional_loss_test.item() if directional_loss_enabled else 0.0,
            total_loss_test.item()
        ]
    }

    loss_summary_df = pd.DataFrame(
        {
            'Metric': final_test_losses["Metric"],
            'Train': final_test_losses["Train"],
            'Test': final_test_losses["Test"]
        }
    ).set_index('Metric').T.rename_axis('Set')

    display(loss_summary_df)
    print()

    # UMAP embedding
    umap_model = umap.UMAP(n_components=2, random_state=42)
    data_2d_umap_train = umap_model.fit_transform(encoded.detach().cpu().numpy())

    train_encodings_umap = pd.DataFrame(data_2d_umap_train, columns = ["UMAP1", "UMAP2"])
    train_encodings_umap["DXGrp"] = y_train_tensor
    train_encodings_umap = pd.merge(train_encodings_umap, train_data_identifiers, left_index=True, right_index=True)


    if return_losses:
        return model,umap_model, train_encodings_umap, encoded.detach().cpu().numpy(), loss_history
    else:

        return model,umap_model, train_encodings_umap, encoded.detach().cpu().numpy()



def test_AE(model, X_test_tensor, y_test_tensor, test_data_identifiers,  umap_model):
    """
    Function to test a trained autoencoder model on the test data, encode the test data, 
    and create embeddings using UMAP (without loss calculations).
    
    Args:
        model (nn.Module): Trained autoencoder model
        X_test_tensor (tensor): Test features (tensor format)
        umap_model (UMAP object): UMAP object used during training for consistent embeddings
    
    Returns:
        model (nn.Module): Trained autoencoder model
        data_2d_umap_test (numpy.ndarray): UMAP embeddings for the test data
        encoded_test (tensor): Encoded features of the test data
    """
    model.eval()

    # Encode the test data
    with torch.no_grad():
        encoded_test = model.encode(X_test_tensor)

    data_2d_umap_test = umap_model.transform(encoded_test.detach().cpu().numpy())

    test_encodings_umap = pd.DataFrame(data_2d_umap_test, columns = ["UMAP1", "UMAP2"])
    test_encodings_umap["DXGrp"] = y_test_tensor
    test_encodings_umap = pd.merge(test_encodings_umap, test_data_identifiers, left_index=True, right_index=True)


    return  test_encodings_umap, encoded_test.detach().cpu().numpy()



class MLPModel(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        self.penultimate = x.clone()  # Penultimate layer activations
        x = self.fc3(x)
        return x


def train_logistic_regression(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, train_data_identifiers, test_data_identifiers, train_data, test_data, plot_subject_progression, compute_violation_ratio_violation_gap):
    """
    Function to train a logistic regression model, generate pseudotime, and compute violation ratios for both train and test data.

    Args:
        X_train_tensor (tensor): Training features
        y_train (array): Training labels
        X_test_tensor (tensor): Test features
        y_test (array): Test labels
        train_data_identifiers (DataFrame): Identifiers for the training data
        test_data_identifiers (DataFrame): Identifiers for the test data
        class_names (list): List of class names
        plot_subject_progression (function): Function to plot subject progression
        compute_violation_ratio_violation_gap (function): Function to compute violation ratios and gaps

    Returns:
        train_df_lr (DataFrame): Training data with pseudotime and logits
        test_df_lr (DataFrame): Test data with pseudotime and logits
    """
    class_names = [f"Class_{cls}" for cls in sorted(train_data["DXGrp"].unique())]


    # Initialize and fit the Logistic Regression model
    finetuned_lr = LogisticRegression(C=100, multi_class='ovr', solver='saga', random_state=42)
    finetuned_lr.fit(X_train_tensor, y_train_tensor)

    weights = finetuned_lr.coef_
    intercepts = finetuned_lr.intercept_

    # Train Dataset: Calculate linear predictor for all classes
    linear_predictor_train = X_train_tensor.detach().numpy().dot(weights.T) + intercepts

    # Create a DataFrame for the logits of each class (Train)
    psuedo_train_lr = pd.DataFrame(linear_predictor_train, columns=[f"Class_{c}_Logit" for c in class_names])

    # Combine with training data identifiers
    train_df_lr = pd.concat([train_data_identifiers, psuedo_train_lr], axis=1)

    # Calculate the maximum logit value for pseudotime (Train)
    train_df_lr['SlingFusedPhatePseu'] = linear_predictor_train.max(axis=1)

    # Normalize the pseudotime (Train)
    train_df_lr['Pseudotime_Normalized'] = (
        (train_df_lr['SlingFusedPhatePseu'] - train_df_lr['SlingFusedPhatePseu'].min()) /
        (train_df_lr['SlingFusedPhatePseu'].max() - train_df_lr['SlingFusedPhatePseu'].min())
    )

    # Test Dataset: Calculate linear predictor for all classes
    linear_predictor_test = X_test_tensor.detach().numpy().dot(weights.T) + intercepts

    # Create a DataFrame for the logits of each class (Test)
    psuedo_test_lr = pd.DataFrame(linear_predictor_test, columns=[f"{c}_Logit" for c in class_names])

    # Combine with test data identifiers
    test_df_lr = pd.concat([test_data_identifiers, psuedo_test_lr], axis=1)

    # Calculate the maximum logit value for pseudotime (Test)
    test_df_lr['SlingFusedPhatePseu'] = linear_predictor_test.max(axis=1)

    # Normalize the pseudotime (Test)
    test_df_lr['Pseudotime_Normalized'] = (
        (test_df_lr['SlingFusedPhatePseu'] - train_df_lr['SlingFusedPhatePseu'].min()) /
        (train_df_lr['SlingFusedPhatePseu'].max() - train_df_lr['SlingFusedPhatePseu'].min())
    )

    # Add diagnostic group information
    train_df_lr["DXGrp"] = y_train_tensor
    test_df_lr["DXGrp"] = y_test_tensor

    return finetuned_lr, train_df_lr, test_df_lr


def train_ENet(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, train_data_identifiers, test_data_identifiers, train_data, test_data, plot_subject_progression, compute_violation_ratio_violation_gap):
    """
    Function to train an Elastic Net regularized logistic regression model, generate pseudotime, and compute violation ratios for both train and test data.

    Args:
        X_train_tensor (tensor): Training features
        y_train (array): Training labels
        X_test_tensor (tensor): Test features
        y_test (array): Test labels
        train_data_identifiers (DataFrame): Identifiers for the training data
        test_data_identifiers (DataFrame): Identifiers for the test data
        class_names (list): List of class names
        plot_subject_progression (function): Function to plot subject progression
        compute_violation_ratio_violation_gap (function): Function to compute violation ratios and gaps

    Returns:
        train_df_en (DataFrame): Training data with pseudotime and logits
        test_df_en (DataFrame): Test data with pseudotime and logits
    """
    class_names = [f"Class_{cls}" for cls in sorted(train_data["DXGrp"].unique())]

    # Initialize and fit the Elastic Net regularized Logistic Regression model
    finetuned_enet_model = LogisticRegression(C=14.149947100260672, l1_ratio=0.15000000000000002,
                                              max_iter=500, multi_class='ovr', penalty='elasticnet',
                                              random_state=42, solver='saga')
    finetuned_enet_model.fit(X_train_tensor, y_train_tensor)

    weights = finetuned_enet_model.coef_
    intercept = finetuned_enet_model.intercept_

    # Train Dataset: Calculating linear predictor (logits)
    logits_train_en = np.dot(X_train_tensor.detach().numpy(), weights.T) + intercept.reshape(1, -1)

    psuedo_train_en = pd.DataFrame(logits_train_en, columns=[f"{c}_Logit" for c in class_names])

    train_df_en = pd.concat([train_data_identifiers, psuedo_train_en], axis=1)

    # Using the maximum logit for pseudotime (Train)
    train_df_en["SlingFusedPhatePseu"] = logits_train_en.max(axis=1)

    # Normalize the pseudotime (Train)
    train_df_en["Pseudotime_Normalized"] = (
        (train_df_en["SlingFusedPhatePseu"] - train_df_en["SlingFusedPhatePseu"].min()) /
        (train_df_en["SlingFusedPhatePseu"].max() - train_df_en["SlingFusedPhatePseu"].min())
    )

    train_df_en["DXGrp"] = y_train_tensor

    logits_test_en = np.dot(X_test_tensor.detach().numpy(), weights.T) + intercept.reshape(1, -1)

    psuedo_test_en = pd.DataFrame(logits_test_en, columns=[f"Class_{c}_Logit" for c in class_names])

    # Combine with test data identifiers
    test_df_en = pd.concat([test_data_identifiers, psuedo_test_en], axis=1)

    # Using the maximum logit for pseudotime (Test)
    test_df_en["SlingFusedPhatePseu"] = logits_test_en.max(axis=1)

    # Normalize the pseudotime (Test)
    test_df_en["Pseudotime_Normalized"] = (
        (test_df_en["SlingFusedPhatePseu"] - train_df_en["SlingFusedPhatePseu"].min()) /
        (train_df_en["SlingFusedPhatePseu"].max() - train_df_en["SlingFusedPhatePseu"].min())
    )

    # Add `DXGrp` for test data
    test_df_en["DXGrp"] = y_test_tensor


    return finetuned_enet_model, train_df_en, test_df_en

def train_MLP(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,train_data_identifiers, test_data_identifiers, train_data, test_data, plot_subject_progression, compute_violation_ratio_violation_gap, seed=42):
    
    class_names = [f"Class_{cls}" for cls in sorted(train_data["DXGrp"].unique())]

    best_params = {'hidden1_size': 32, 'hidden2_size': 256, 'learning_rate': 0.01, 'loss_function': 'Huber'}

    set_global_seed(seed)

    mlp_best_model = MLPModel(X_train_tensor.shape[1], best_params['hidden1_size'], best_params['hidden2_size'])
    best_criterion = nn.SmoothL1Loss()
    best_optimizer = torch.optim.Adam(mlp_best_model.parameters(), lr=best_params['learning_rate'])

    mlp_best_model.train()
    for epoch in range(15):
        best_optimizer.zero_grad()
        output = mlp_best_model(X_train_tensor)
        loss = best_criterion(output, y_train_tensor)
        loss.backward()
        best_optimizer.step()

    # Evaluate on both training and test sets
    mlp_best_model.eval()

    # For training set
    with torch.no_grad():
        train_output = mlp_best_model(X_train_tensor)
        train_penultimate_activations = mlp_best_model.penultimate
        train_final_weights = mlp_best_model.fc3.weight
        train_final_bias = mlp_best_model.fc3.bias
        train_weighted_sum = torch.matmul(train_penultimate_activations, train_final_weights.t()) + train_final_bias

    # For test set
    with torch.no_grad():
        test_output = mlp_best_model(X_test_tensor)
        test_penultimate_activations = mlp_best_model.penultimate
        test_final_weights = mlp_best_model.fc3.weight
        test_final_bias = mlp_best_model.fc3.bias
        test_weighted_sum = torch.matmul(test_penultimate_activations, test_final_weights.t()) + test_final_bias


    pseudo_train_mlp = pd.DataFrame(train_weighted_sum.numpy(), columns=["SlingFusedPhatePseu"])
    psuedo_test_mlp = pd.DataFrame(test_weighted_sum.numpy(), columns=["SlingFusedPhatePseu"])


    train_df_mlp = pd.concat([test_data_identifiers, pseudo_train_mlp], axis=1)
    test_df_mlp = pd.concat([test_data_identifiers, psuedo_test_mlp], axis=1)

    train_df_mlp['Pseudotime_Normalized'] = ((train_df_mlp['SlingFusedPhatePseu'] - train_df_mlp['SlingFusedPhatePseu'].min()) /
                                              (train_df_mlp['SlingFusedPhatePseu'].max() - train_df_mlp['SlingFusedPhatePseu'].min()))
    train_df_mlp["DXGrp"] = y_train_tensor.numpy()  # Add diagnosis group to train data

    # Process test data (Normalized Logit)
    test_df_mlp['Pseudotime_Normalized'] = ((test_df_mlp['SlingFusedPhatePseu'] - train_df_mlp['SlingFusedPhatePseu'].min()) /
                                            (train_df_mlp['SlingFusedPhatePseu'].max() - train_df_mlp['SlingFusedPhatePseu'].min()))
    test_df_mlp["DXGrp"] = y_test_tensor.numpy()  


    return mlp_best_model, train_df_mlp, test_df_mlp


def count_dropping_subjects(data, rid_col, age_col, pseudotime_col):
    """
    Identifies the number of subjects with reducing pseudotime.

    Parameters:
    - data: pd.DataFrame - Input dataset.
    - rid_col: str - Column name for subject ID (e.g., "RID").
    - age_col: str - Column name for age (e.g., "AGE").
    - pseudotime_col: str - Column name for pseudotime (e.g., "Pseudotime_Normalized").

    Returns:
    - int: Number of unique subjects with reducing pseudotime.
    """
    sorted_data = data.sort_values([rid_col, age_col])
    rid_counts = sorted_data[rid_col].value_counts().reset_index(name="count")
    repeated_rid_data = data[
        data[rid_col].isin(rid_counts[rid_counts["count"] > 1]["RID"].values)
    ].sort_values([rid_col, age_col])
    repeated_rid_data["Reduced"] = repeated_rid_data.groupby(rid_col)[pseudotime_col].diff().lt(0)
    reducing_rids = repeated_rid_data[repeated_rid_data["Reduced"]][rid_col].unique()
    return len(reducing_rids)


def save_pickle_object(filename, model_obj):
    with open(filename, "wb") as file:
        pickle.dump(model_obj, file)

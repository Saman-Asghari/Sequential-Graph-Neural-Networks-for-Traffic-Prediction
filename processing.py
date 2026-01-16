import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_and_prepare_data(input_seq_length,batch_size,device,train_portion=0.85,data_dir="./dataset"):

    try:
        node_static = np.load(f"{data_dir}/node_static.npy", allow_pickle=True)
        edge_static = np.load(f"{data_dir}/edge_static.npy", allow_pickle=True)
        edge_indices = np.load(f"{data_dir}/edge_indices.npy", allow_pickle=True)
        speed_data = np.load(f"{data_dir}/speed_data.npy", allow_pickle=True)
    except FileNotFoundError:
        raise Exception("Data files not found! Please run the processing script first.")

   
    speed_data = speed_data[:, :, 1:, :]
    speed_data = speed_data.astype(np.float32)

          
    if speed_data.shape[2] != node_static.shape[0]:
        raise ValueError("Mismatch in number of nodes between speed_data and node_static after removing timestamp node.")
        

    node_static = node_static.astype(np.float32)
    edge_static = edge_static.astype(np.float32)
    edge_indices = edge_indices.astype(np.int64)

    print(f"   - Final Data Shape: {speed_data.shape}")

    edge_index_tensor = torch.LongTensor(edge_indices).to(device)
    node_static_tensor = torch.FloatTensor(node_static).to(device)
    edge_static_tensor = torch.FloatTensor(edge_static).to(device)

    total_samples = speed_data.shape[0]
    train_size = int(total_samples * train_portion)

    train_data = speed_data[:train_size]
    test_data = speed_data[train_size:]

    X_train = torch.FloatTensor(train_data[:, :input_seq_length, :, :])
    Y_train = torch.FloatTensor(train_data[:, input_seq_length:, :, :])

    X_test = torch.FloatTensor(test_data[:, :input_seq_length, :, :])
    Y_test = torch.FloatTensor(test_data[:, input_seq_length:, :, :])
    # DataLoaders
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        test_loader,
        X_train,
        Y_train,
        X_test,
        Y_test,
        node_static_tensor,
        edge_static_tensor,
        edge_index_tensor
    )
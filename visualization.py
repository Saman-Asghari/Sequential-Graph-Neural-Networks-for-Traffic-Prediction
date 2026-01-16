import numpy as np
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import torch
from typing import Callable, Optional

def plot_traffic_graph_streetlike(
    DATA_DIR="./dataset",
    window_size=36,
    stride=12,
    eliminate_self_edges=True,
    window_index=0,
    timestep_index=-1,
    use_geo_if_available=True,
    label_k=25,
    figsize=(28, 24),
    cmap=plt.cm.coolwarm,
    edge_alpha=0.55,
    edge_width=3,
    node_size=1000,
    arrows=False,
):
  
    try:
        df = pd.read_hdf(os.path.join(DATA_DIR, "metr-la.h5"))
    except Exception:
        df = pd.read_csv(os.path.join(DATA_DIR, "metr-la.csv"))

    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    df = df.select_dtypes(include=[np.number])

    data = df.values 
    data = np.expand_dims(data, axis=-1) 

    windows = []
    for i in range(0, len(data) - window_size, stride):
        windows.append(data[i : i + window_size])
    speed_data = np.array(windows)  # (num_windows, window_size, num_nodes, 1)

    if speed_data.shape[0] == 0:
        raise ValueError("No windows created. Check window_size/stride vs data length.")
    if not (0 <= window_index < speed_data.shape[0]):
        raise IndexError(f"window_index out of range: 0..{speed_data.shape[0]-1}")

    node_attrs = np.load(os.path.join(DATA_DIR, "node_static.npy"), allow_pickle=True)
    edge_attrs = np.load(os.path.join(DATA_DIR, "edge_static.npy"), allow_pickle=True)
    edge_indices = np.load(os.path.join(DATA_DIR, "edge_indices.npy"), allow_pickle=True)

    num_nodes = node_attrs.shape[0]
    num_edges = edge_indices.shape[1]

    current_speed = speed_data[window_index, timestep_index, :, 0].astype(float)
    current_speed = current_speed[:num_nodes]  # safety truncate

    G = nx.DiGraph()
    for i in range(num_nodes):
        deg = float(node_attrs[i, 0]) if node_attrs.ndim > 1 else float(node_attrs[i])
        G.add_node(i, degree=deg, speed=float(current_speed[i]))

    for i in range(num_edges):
        src, dst = int(edge_indices[0, i]), int(edge_indices[1, i])
        weight = float(edge_attrs[i][0])  # edge_attrs row looks like [weight, ...]
        if (not eliminate_self_edges) or (src != dst):
            G.add_edge(src, dst, weight=weight)

    pos = None
    if use_geo_if_available:
        candidates = [
            "graph_sensor_locations.csv",
            "sensor_locations.csv",
            "metr-la_sensor_locations.csv",
            "locations.csv",
        ]
        loc_path = None
        for fn in candidates:
            p = os.path.join(DATA_DIR, fn)
            if os.path.exists(p):
                loc_path = p
                break

        if loc_path is not None:
            loc = pd.read_csv(loc_path)
            cols = {c.lower(): c for c in loc.columns}
            lat_col = cols.get("latitude") or cols.get("lat")
            lon_col = cols.get("longitude") or cols.get("lon") or cols.get("lng")
            id_col = cols.get("sensor_id") or cols.get("id") or cols.get("sid") or cols.get("index")

            if lat_col and lon_col:
                if id_col:
                    id_to_row = {int(r[id_col]): r for _, r in loc.iterrows()}
                    pos = {
                        i: (float(id_to_row[i][lon_col]), float(id_to_row[i][lat_col]))
                        for i in G.nodes
                        if i in id_to_row
                    }
                else:
                    pos = {
                        i: (float(loc.loc[i, lon_col]), float(loc.loc[i, lat_col]))
                        for i in G.nodes
                        if i < len(loc)
                    }

        if pos is not None and len(pos) < 0.9 * G.number_of_nodes():
            pos = None

    if pos is None:
        
        for u, v, d in G.edges(data=True):
            w = float(d.get("weight", 1.0))
            d["dist"] = 1.0 / (w + 1e-6)
        pos = nx.kamada_kawai_layout(G, weight="dist")
        
        

    fig, ax = plt.subplots(figsize=figsize)
   
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_width,
        alpha=edge_alpha,
        arrows=arrows,
        node_size=node_size,
        

    )

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=current_speed,
        node_size=node_size,
        cmap=cmap,
        linewidths=0.0,
    )

    labels = {i: f"{current_speed[i]:.1f}" for i in range(num_nodes)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
    

    cbar = fig.colorbar(nodes, ax=ax, label="Traffic Speed")
    ax.set_title("Traffic Network (street-like) with Node Speeds")
    ax.set_axis_off()
    ax.set_aspect("equal")

    print(f"nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    print(f"mean speed {current_speed.mean():.2f}, lowest speed: {current_speed.min():.2f}, max speed: {current_speed.max():.2f}")

    plt.show()


@torch.no_grad()
def show_prediction_with_history(
    model,
    test_loader,
    device,
    prepare_fn: Callable,
    sample_id: int = 0,   
    node_id: int = 0,
    feature_in: int = 0,  
    title: str = "Prediction (Past + Future)",
    predict_fn: Optional[Callable] = None,
):
    model.eval()


    ds = test_loader.dataset
    X, Y = ds[sample_id]          
    X = X.unsqueeze(0).to(device) 
    Y = Y.unsqueeze(0).to(device) 

    X_flat, Y_flat = prepare_fn(X, Y)
    if predict_fn is None:
        Y_pred = model(X_flat)
    else:
        Y_pred = predict_fn(model, X_flat, Y_flat)

    B, Tin, N, Fin = X.shape
    idx = 0 * N + node_id  # B=1

    hist = X[0, :, node_id, feature_in].detach().cpu().numpy().squeeze()

    gt = Y_flat[idx].detach().cpu().numpy().squeeze()
    pr = Y_pred[idx].detach().cpu().numpy().squeeze()

    hist = np.ravel(hist)
    gt   = np.ravel(gt)
    pr   = np.ravel(pr)

    assert gt.shape == pr.shape, (gt.shape, pr.shape)

    
    t_past = np.arange(-len(hist), 0)
    t_fut  = np.arange(0, len(gt))

    plt.figure(figsize=(8, 4))
    plt.plot(t_past, hist, marker=".", label="History (input)")
    plt.plot(t_fut, gt, marker="o", label="Ground Truth (future)")
    plt.plot(t_fut, pr, marker="x", label="Prediction (future)")

    plt.axvline(-0.5, linestyle="--")  # split line between past and future
    plt.xlabel("Time step (past → future)")
    plt.ylabel("Value")
    plt.title(f"{title} | sample={sample_id}, node={node_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
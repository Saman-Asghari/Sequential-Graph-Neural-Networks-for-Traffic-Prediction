# metr_la_prep.py
from __future__ import annotations

import os
import subprocess
import pickle
from dataclasses import dataclass
import subprocess
from typing import Tuple, Optional
import wget
import numpy as np
import pandas as pd
from zmq import proxy


@dataclass(frozen=True)
class MetrLAConfig:
    data_dir: str = "./dataset"

    adj_url: str = "https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/adj_mx.pkl"
    locations_url: str = (
        "https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/graph_sensor_locations.csv"
    )

    speed_url: str = "https://zenodo.org/record/5146275/files/METR-LA.csv?download=1"

    input_len: int = 24
    output_len: int = 12
    stride: int = 12  # 12 * 5min = 60min (1 hour) if data is 5-min intervals


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _download_file(url: str, out_path: str,proxy: Optional[str] = None,force_download=False) -> str:
   
    _ensure_dir(os.path.dirname(out_path) or ".")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and not force_download:
        print(f"File already exists: {out_path}")
        return out_path
    if( proxy is None):
        wget.download(url, out=out_path)
    else:
        cmd = [
            "curl",
            "-L",                     
            "-o", out_path,
            "--proxy", proxy,
            url,
        ]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as e:
            raise RuntimeError("curl is not installed or not in PATH.") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"curl download failed for {url}") from e
    return out_path

def download_metr_la_assets(cfg: MetrLAConfig,proxy: Optional[str] = None,force_download=False) -> dict:
    """
    Downloads required files into cfg.data_dir.
    Returns dict of local file paths.
    """
    _ensure_dir(cfg.data_dir)

    adj_path = os.path.join(cfg.data_dir, "adj_mx.pkl")
    loc_path = os.path.join(cfg.data_dir, "graph_sensor_locations.csv")
    csv_path = os.path.join(cfg.data_dir, "metr-la.csv")

    print("DOWNLOADING " + "=" * 60)
    _download_file(cfg.adj_url, adj_path,proxy=proxy,force_download=force_download)
    _download_file(cfg.locations_url, loc_path,proxy=proxy,force_download=force_download)
    _download_file(cfg.speed_url, csv_path,proxy=proxy,force_download=force_download)

    return {"adj": adj_path, "locations": loc_path, "speed_csv": csv_path}




def save_npy_file(data, file_path,force_process):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0 and not force_process:
        print(f"File already exists: {file_path}")
    else:
        np.save(file_path, data)
        print(f"Saved file: {file_path}")

def prepare_metr_la_dataset(cfg: Optional[MetrLAConfig] = None, proxy: Optional[str] = None,force_download=False,force_process=False) -> dict:
   
    cfg = cfg or MetrLAConfig()
    paths = download_metr_la_assets(cfg, proxy=proxy,force_download=force_download)

    print("PROCESSING " + "=" * 60)
    
    with open(f"{cfg.data_dir}/adj_mx.pkl", "rb") as f:
        try:
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")
        except:
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f)

    if isinstance(adj_mx, np.ndarray):
        rows, cols = np.where(adj_mx > 0)
        values = adj_mx[rows, cols]
    else:
        # if it's scipy sparse
        adj_mx = adj_mx.tocoo()
        rows, cols = adj_mx.row, adj_mx.col
        values = adj_mx.data

    edge_indices = np.array([rows, cols])
    num_nodes = adj_mx.shape[0]
    print(f"   - Graph: {num_nodes} Nodes, {len(rows)} Edges.")

    try:
        if os.path.exists(f"{cfg.data_dir}/metr-la.h5"):
            df = pd.read_hdf(f"{cfg.data_dir}/metr-la.h5")
        else:
            df = pd.read_csv(f"{cfg.data_dir}/metr-la.csv")
            if "cost" in df.columns:
                df = df.drop(columns=["cost"])  # clean garbage
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
    except Exception as e:
        print(f"   Error reading data files: {e}")
        print("   Using SYNTHETIC data as fallback so you can run code.")
        df = pd.DataFrame(np.random.rand(2000, num_nodes) * 60)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill().bfill()
    if df.isna().any().any():
        df = df.fillna(0.0)

    data = df.values.astype(np.float32)  # Shape (T, N)
    data = np.expand_dims(data, axis=-1)  # Shape (T, N, 1)

    steps_per_day = 288
    num_days = data.shape[0] // steps_per_day

    data = data[: num_days * steps_per_day]

    print("   - Reshaping into 3-hour windows (Input=2h, Output=1h)...")
    window_size = 36  
    stride = 12      

    windows = []
    for i in range(0, len(data) - window_size, stride):
        windows.append(data[i : i + window_size])

    speed_data = np.array(windows, dtype=np.float32)  

    node_degree = (
        np.sum(adj_mx > 0, axis=1).A1 if hasattr(adj_mx, "A1") else np.sum(adj_mx > 0, axis=1)
    )
    node_degree = (node_degree - node_degree.min()) / (node_degree.max() + 1e-5)

    node_attrs = np.zeros((num_nodes, 4), dtype=np.float32)
    node_attrs[:, 0] = node_degree

    edge_attrs = np.zeros((len(rows), 2), dtype=np.float32)
    edge_attrs[:, 0] = np.asarray(values, dtype=np.float32)



    save_npy_file(node_attrs, f"{cfg.data_dir}/node_static.npy",force_process)
    save_npy_file(edge_attrs, f"{cfg.data_dir}/edge_static.npy",force_process)
    save_npy_file(edge_indices, f"{cfg.data_dir}/edge_indices.npy",force_process)
    save_npy_file(speed_data, f"{cfg.data_dir}/speed_data.npy",force_process)
    

    return {
        "data_dir": cfg.data_dir,
        "paths": {
            **paths,
            "node_static": os.path.join(cfg.data_dir, "node_static.npy"),
            "edge_static": os.path.join(cfg.data_dir, "edge_static.npy"),
            "edge_indices": os.path.join(cfg.data_dir, "edge_indices.npy"),
            "speed_data": os.path.join(cfg.data_dir, "speed_data.npy"),
        },
        "shapes": {
            "speed_data": tuple(speed_data.shape),
            "node_static": tuple(node_attrs.shape),
            "edge_static": tuple(edge_attrs.shape),
            "edge_indices": tuple(edge_indices.shape),
        },
        "config": cfg,
    }

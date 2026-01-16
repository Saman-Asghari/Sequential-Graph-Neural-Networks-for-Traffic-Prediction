# Sequential-Graph-Neural-Networks-for-Traffic-Prediction

Short-term traffic speed forecasting on the METR-LA sensor network, comparing a custom **Sequential Graph Neural Network (SeqGNN)** against four strong temporal baselines: MLP, Nested LSTM (NLSTM), Seq2Seq LSTM, and a Dual-Stage Attention RNN (DA-RNN).

Given 2 hours of past traffic speeds (24 steps at 5-minute intervals) across ~207 road sensors, the models predict the next 1 hour (12 steps). SeqGNN represents the road network explicitly as a graph — sensors as nodes, road connectivity as directed edges — and models congestion propagation with a Graph Recurrent Neural Network (GRNN), rather than treating each sensor as an independent time series.

📄 Full writeup with methodology, related work, and analysis: [`final_report.pdf`](./final_report.pdf)

## Overview

Traffic states evolve dynamically and propagate through network *connectivity*, not just time or spatial proximity. Classical time-series models (LSTMs, MLPs) miss this relational structure. SeqGNN addresses this by treating both the input and output as **sequences of attributed graphs**:

- **Nodes** carry traffic speed + static features (e.g. degree).
- **Edges** carry static connectivity strength.
- A **GN Block** (Graph Network block) updates edge, node, and global features via message passing.
- A **GRNN Block** wraps the GN block with recurrent hidden state to model temporal dynamics.
- An encoder consumes the historical graph sequence; a decoder autoregressively generates the future graph sequence.

## Results

Evaluated on METR-LA (207 sensors, ~1,500 edges, 5-minute sampling, 24-step input → 12-step output):

| Model              | MAE      | RMSE      |
|---------------------|:--------:|:---------:|
| MLP                  | 5.23     | 12.59     |
| NLSTM                | 5.09     | 12.57     |
| Seq2Seq              | 5.08     | 12.69     |
| DA-RNN               | 5.15     | 12.75     |
| **SeqGNN (proposed)**| **5.08** | **12.11** |

SeqGNN achieves the best RMSE and ties for the best MAE, and qualitatively captures downstream congestion propagation along road connectivity that purely temporal models miss — though the margin over the strongest baselines is smaller than expected, likely due to the static-graph assumption and the relatively short forecast horizon. See `final_report.pdf` for full discussion, limitations, and future work.

## Project Structure

```
graph/
├── main.ipynb                 # End-to-end notebook: data prep, training, evaluation, plots for all models
├── final_report.pdf           # Full project writeup
└── src/
    ├── metr_la_prep.py        # Downloads & preprocesses the METR-LA dataset into graph tensors
    ├── processing.py          # Loads processed .npy files into train/test DataLoaders
    ├── visualization.py       # Network map plotting + prediction visualization
    └── models/
        ├── mlp.py              # MLP baseline (per-sensor, independent)
        ├── NLSTM.py            # Nested LSTM baseline
        ├── Seq2Seq.py          # Encoder-decoder LSTM baseline
        ├── DARNN.py            # Dual-stage attention RNN baseline
        └── SeqGNN.py           # Proposed Graph Network + GRNN model
```

## Dataset

Uses the **METR-LA** traffic speed dataset (Los Angeles highway loop detectors), assembled from three public sources at runtime:

- Speed readings — Zenodo mirror of METR-LA
- Sensor adjacency matrix — from the [DCRNN](https://github.com/liyaguang/DCRNN) repository
- Sensor lat/lon locations — from the same DCRNN repository

`src/metr_la_prep.py` downloads these automatically into `./dataset` on first run, builds the sensor graph (node/edge attributes, edge index), reshapes the speed series into sliding windows, and caches everything as `.npy` files so subsequent runs skip re-downloading/re-processing.

## Implemented Paper
you can see the actual paper from [paper](https://ieeexplore.ieee.org/document/8708297)
## Authors

Saman Asghari, Erfan Geramizadeh — February 2026

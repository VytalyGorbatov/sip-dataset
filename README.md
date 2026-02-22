# SIP Dataset

This repository contains a dataset of SIP (Session Initiation Protocol) traffic, categorized into benign and attack samples primarily for training machine learning models for intrusion detection or traffic analysis.

## Structure

The dataset is organized into two main directories:

-   `attack/`: Contains malicious traffic samples.
    -   `attack_train.json`
    -   `attack_val.json`
    -   `attack_test.json`
-   `benign/`: Contains normal traffic samples.
    -   `benign_train.json`
    -   `benign_val.json`
    -   `benign_test.json`

## Data Format

The files are in JSON format. Each file contains a root object with a `dataset` key, which holds a list of flow records.

### Global Fields

Each file contains the following global mapping fields that define how categorical values are encoded:

-   `proto_mapping`: Maps protocol names to integers (e.g., `{"UDP": 0}`).
-   `pkt_gen_mapping`: Maps packet generation types to integers (e.g., `{"raw": 0, "stream_ip": 1}`).
-   `eth_type_mapping`: Maps Ethernet types to integers (e.g., `{"0x800": 0}`).
-   `dir_mapping`: Maps traffic direction to integers (e.g., `{"C2S": 0}`).

### Schema

Each record in the `dataset` array contains the following fields:

-   `flowstart_time`: Timestamp of the flow start.
-   `seconds`: Duration or related time metric.
-   `proto`: Protocol identifier (e.g., UDP).
-   `pkt_gen`: Packet generation type.
-   `pkt_len`: Packet length.
-   `eth_len`: Ethernet frame length.
-   `eth_type`: Ethernet type.
-   `ip_id`: IP identification number.
-   `ip_len`: IP packet length.
-   `tos`: Type of Service.
-   `ttl`: Time to Live.
-   `udp_len`: UDP length.
-   `dir`: Direction of traffic.
-   `client_bytes`: Bytes sent by the client.
-   `client_pkts`: Packets sent by the client.
-   `server_bytes`: Bytes sent by the server.
-   `server_pkts`: Packets sent by the server.
-   `buffers`: The raw content of the SIP message (headers and body).
-   `buffer_names`: Names of the buffer sections (e.g., `sip_header`, `sip_body`).
-   `alerted`: Indicator if the flow raised an alert (0 or 1).
-   `is_attack`: Label indicating if the flow is an attack (1) or benign (0).

## Usage

You can load the dataset using standard JSON libraries in Python or other languages.

```python
import json

def load_data(filepath):
    with open(filepath, 'r') as f:
```

## Scripts

This repository includes several utility scripts for data processing and analysis.

### Preprocessors (`preprocessors/`)

-   **`process_json_dataset.py`**:
    -   Builds the dataset with deduplication and normalization.
    -   Adds the `is_attack` flag to records.
    -   Handles categorical mappings.
    -   Usage: `python preprocessors/process_json_dataset.py -i <input.json> -o <output.json> --is-attack`

-   **`split_json_dataset.py`**:
    -   Splits a JSON dataset into training, validation, and test subsets.
    -   Supports custom split ratios and random seeding.
    -   Usage: `python preprocessors/split_json_dataset.py <dataset.json> --splits 0.7 0.15 0.15`

### Postprocessors (`postprocessors/`)

-   **`analyze_buffers.py`**:
    -   Analyzes the raw bytes of the SIP message `buffers`.
    -   Useful for inspecting payload content distributions.

-   **`feature_spread.py`**:
    -   Visualizes the distribution and spread of various features in the dataset.
    -   Uses Pandas and Seaborn for data analysis and plotting.

-   **`heatmap_corr.py`**:
    -   Generates correlation heatmaps to visualize relationships between features.
    -   Helps identify redundant or highly correlated attributes.     data = json.load(f)
    return data['dataset']

# Example usage
train_benign = load_data('benign/benign_train.json')
train_attack = load_data('attack/attack_train.json')

print(f"Loaded {len(train_benign)} benign records.")
print(f"Loaded {len(train_attack)} attack records.")
```

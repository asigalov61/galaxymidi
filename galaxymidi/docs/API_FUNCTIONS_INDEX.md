# galaxymidi — API Functions Index

Quick index of all public functions in the **galaxymidi** package.
For detailed parameter/return documentation, see [`API_REFERENCE.md`](./API_REFERENCE.md).

| | |
|---|---|
| **Package** | galaxymidi |
| **Version** | 1.0.0 |
| **Modules** | `galaxymidi` (main), `galaxymidi.helpers` |
| **License** | Apache License 2.0 |

## Installation

```bash
pip install midisimx
```

## Quick Start

```python
import galaxymidi

galaxymidi.download_dataset()               # Fetch dataset archive from Hugging Face Hub
galaxymidi.parallel_extract()               # Extract archive to disk (parallel writes)

features   = galaxymidi.load_features()     # Precomputed feature store (LMBD)
embeddings = galaxymidi.load_embeddings()   # Precomputed embeddings (memory-mapped)

from galaxymidi import helpers

info = helpers.get_normalized_midi_md5_hash('song.mid')   # Canonical MIDI fingerprint
helpers.install_apt_package('fluidsynth')                 # System package (Linux/Debian)
```

---

## Functions Index — `galaxymidi` (main module)

| Function | Description | Details |
|---|---|---|
| `download_dataset` | Download the dataset archive from the Hugging Face Hub. | [→](./API_REFERENCE.md#download_dataset) |
| `parallel_extract` | Extract the dataset `.tar.gz` archive using parallel disk writes. | [→](./API_REFERENCE.md#parallel_extract) |
| `load_features` | Load the precomputed features file (LMBD format). | [→](./API_REFERENCE.md#load_features) |
| `load_embeddings` | Load the precomputed embeddings as a paired memory map. | [→](./API_REFERENCE.md#load_embeddings) |
| `extract_midi_features` | Extract a structured feature dictionary from a MIDI file. | [→](./API_REFERENCE.md#extract_midi_features) |
| `render_midi` | Render a MIDI file to WAV via a SoundFont. | [→](./API_REFERENCE.md#render_midi) |
| `read_jsonl` | Read a JSONL file into a list of records. | [→](./API_REFERENCE.md#read_jsonl) |
| `write_file` | *(Internal)* Write bytes to a file, creating parent directories. | [→](./API_REFERENCE.md#write_file) |

## Functions Index — `galaxymidi.helpers`

| Function | Description | Details |
|---|---|---|
| `sort_aligned_lists` | Sort two aligned lists in place by MIDI sequence length. | [→](./API_REFERENCE.md#sort_aligned_lists) |
| `get_normalized_midi_md5_hash` | Compute original and normalized MD5 hashes for a MIDI file. | [→](./API_REFERENCE.md#get_normalized_midi_md5_hash) |
| `normalize_midi_file` | Normalize a MIDI file and write it to disk. | [→](./API_REFERENCE.md#normalize_midi_file) |
| `is_installed` | Check whether a Debian package is installed (`dpkg-query`). | [→](./API_REFERENCE.md#is_installed) |
| `install_apt_package` | Install an apt package idempotently, with retries. | [→](./API_REFERENCE.md#install_apt_package) |
| `_run_apt_get` | *(Internal)* Run an `apt-get` command with forced-config options. | [→](./API_REFERENCE.md#_run_apt_get) |

---

## Module Attributes

| Attribute | Value |
|---|---|
| `galaxymidi.__version__` | `'1.0.0'` |
| `galaxymidi.helpers.__version__` | `'1.0.0'` |

## Import Notes

- Both modules print a loading banner to stdout on import.
- The main module sets `HF_XET_HIGH_PERFORMANCE=1` for faster Hugging Face transfers.
- `galaxymidi.helpers` requires **Python 3.9+** (PEP 585 `list[str]` annotations).
- `is_installed` / `install_apt_package` are **Debian/Ubuntu (Linux) only**.

---

### Project Los Angeles
### Tegridy Code 2026
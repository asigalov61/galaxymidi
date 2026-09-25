# Galaxy MIDI Dataset
## The largest and most comprehensive MIDI dataset in the known galaxy

<img width="1536" height="1024" alt="Galaxy-MIDI-Dataset" src="https://github.com/user-attachments/assets/a4f9fc8f-ee76-4b25-b6bf-9b83010156da" />

***

![License](https://img.shields.io/badge/License-CC--BY--NC--SA_4.0-lightgrey)
![Size](https://img.shields.io/badge/Size-152_GB-blue)
![MIDI files](https://img.shields.io/badge/MIDI_files-19.8M-green)

***

> **19,893,571 unique, standardized MIDI files with pre-computed features, midisimx embeddings, lyrics, karaoke data, and genre labels, ready for Music AI and MIR.**

***

## Contents

- [Introduction](#introduction)
- [Abstract](#abstract)
- [Overview](#overview)
- [Key features](#key-features)
- [Installation and use](#installation-and-use)
- [Dataset structure](#dataset-structure)
- [Dataset statistics](#dataset-statistics)
- [Dataset integrity verification info](#dataset-integrity-verification-info)
- [License](#license)
- [Citations](#citations)
- [Acknowledgments](#acknowledgments)

***

## Introduction

The **Galaxy MIDI Dataset** is the flagship symbolic music corpus from **Project Los Angeles / Tegridy Code** — the direct continuation of a line of record-setting open datasets that includes the Tegridy MIDI Dataset and the Discover MIDI Dataset (see [Citations](#citations)).

At its core are **19,893,571 unique, standardized, and original MIDI files** — but Galaxy goes far beyond raw MIDI. Every portion of the dataset is enriched with pre-computed data: comprehensive MIDI features, embeddings, file identifications, genre/artist/song labels, karaoke tracks, aligned English lyrics, curated subsets, and high-quality soundfonts for rendering.

Working with a dataset of this scale is made effortless by the companion **`galaxymidi`** Python package: downloading and extracting the entire 152GB corpus takes just a few lines of code. The embeddings portion of the dataset is powered by **`midisimx`**, which `galaxymidi` depends on and installs automatically.

The entire dataset is released under the **CC BY-NC-SA 4.0** license, making it free for non-commercial research, education, and creative use with attribution.

***

## Abstract

We present the Galaxy MIDI Dataset — to our knowledge, the largest and most comprehensive symbolic music dataset ever released. The dataset consists of 19,893,571 unique, standardized, and original MIDI files (152 GB), accompanied by rich pre-computed annotations and derived data: 17,190,286 comprehensive MIDI feature sets, 15,591,559 embeddings computed with the midisimx embedding model, 9,361,365 file identifications, 89,569 files with genre, artist, and song labels, 99,475 karaoke MIDIs, and 47,069 files with corresponding English lyrics. Curated subsets include 2,111,480 MIDIs with monophonic melodies, 754,092 solo melody MIDIs, and 586,398 solo drum-track MIDIs. The dataset ships with eight curated JSONL file lists for memory-efficient subsetting, four production-grade soundfonts for rendering and auditioning, and full integrity verification via MD5 and SHA256 checksums. The entire corpus is released under CC BY-NC-SA 4.0 together with a lightweight companion Python package (galaxymidi) that handles downloading and fast parallel extraction in a few lines of code. The Galaxy MIDI Dataset is designed to support music information retrieval (MIR), symbolic music generation, music understanding, representation learning, and large-scale pretraining for Music AI.

***

## Overview

The dataset is organized into a small number of self-describing top-level directories:

| Directory / file | Description | Scale |
| :--- | :--- | :--- |
| `MIDIs` | The complete corpus of unique, standardized, original MIDI files | 19,893,571 files |
| `Features` | Pre-computed comprehensive MIDI features, merged into a single container | 17,190,286 entries |
| `Embeddings` | Pre-computed midisimx embeddings for the clean MIDI subset | 15,591,559 vectors |
| `Files Lists` | Curated JSONL file lists for fast subsetting and streaming | 8 lists |
| `Identified` | MIDI file identifications | 9,361,365 entries |
| `Genres` | Genre, artist, and song labels | 89,569 MIDIs |
| `Karaoke` | Karaoke MIDI data | 99,475 MIDIs |
| `Lyrics` | Aligned English lyrics | 47,069 MIDIs |
| `SoundFonts` | High-quality SF2 soundfonts for rendering and auditioning | 4 soundfonts |
| `Artwork` | Official dataset artwork | — |
| `Code` | Bundled source code: `midisimx`, `tegridy-tools`, and license | — |
| `ATTRIBUTION.txt` | Source attribution information | — |

The curated **Files Lists** provide instant access to all major subsets without scanning the full corpus: `all`, `clean`, `aligned`, `drum_track`, `genres`, `karaoke`, `mono_melodies`, and `solo_melodies`.

**Tooling**

* [`galaxymidi`](https://pypi.org/project/galaxymidi/) — companion pip package for one-command download, fast parallel extraction, and data loading. Installable via `pip install galaxymidi`.
* [`midisimx`](https://pypi.org/project/midisimx/) — MIDI similarity and embedding model used to compute the pre-computed embeddings portion of the dataset. Installed automatically as a dependency of `galaxymidi`.

***

## Key features

* 🌌 **Enormous scale** — 19,893,571 unique MIDI files / 152GB: the largest MIDI dataset of its kind
* 🎼 **Unique, standardized, and original** — fully deduplicated and standardized corpus
* 🎛️ **17,190,286 pre-computed comprehensive MIDI features** — no extraction pipeline required
* 🧬 **15,591,559 pre-computed midisimx embeddings** — ready for similarity search, retrieval, and downstream modeling
* 🔎 **9,361,365 identified MIDIs** — matched and identified files
* 🎤 **99,475 karaoke MIDIs** and **47,069 MIDIs with aligned English lyrics**
* 🏷️ **89,569 MIDIs with genre, artist, and song labels**
* 🎹 **Rich curated subsets** — 2,111,480 mono-melody MIDIs, 754,092 solo melodies, 586,398 solo drum tracks
* 📋 **8 curated JSONL file lists** — instant, memory-efficient subsetting
* 🔊 **4 production-grade soundfonts** for immediate rendering and auditioning
* 🧰 **Companion pip package** — download and extract the whole dataset in two lines of code
* ✅ **Full integrity verification** — MD5 and SHA256 checksums included
* ⚖️ **Clearly licensed** — CC BY-NC-SA 4.0

***

## Installation and use

**Requirements:** Python 3.x with pip. Please note that the dataset size is **152GB** compressed, so make sure you have sufficient disk space and a stable internet connection. A multi-core machine is recommended for fast parallel extraction.
**Dependencies** Please see official [midisimx](https://github.com/asigalov61/midisimx) repo for detailed information

### 1) Install

```sh
# Install easily with pip
pip install -U galaxymidi
```

### 2) Download

```python
# Download the dataset from Hugging Face

# Please note that the dataset size is 152GB
# so it make take a while to download
# depending on your internet speed
import galaxymidi

galaxymidi.download_dataset()
```

Alternatively, download directly with the Hugging Face CLI:

```sh
pip install -U "huggingface_hub[cli]"
huggingface-cli download projectlosangeles/Galaxy-MIDI-Dataset --repo-type dataset --local-dir Galaxy-MIDI-Dataset
```

### 3) Extract

```python
# Extract fast as follows
from galaxymidi import fast_parallel_extract

fast_parallel_extract.fast_parallel_extract()
```

### 4) Quickstart

```python
import galaxymidi

# Load one of the curated file lists
galaxymidi.load_jsonl('Files Lists/clean_midis_files_list.jsonl)
```

```python
import midisimx

# Load the pre-computed Galaxy embeddings (midisimx format)
embeddings = galaxymidi.load_embeddings(Embeddings/galaxy_midi_dataset_embeddings_1_2_1_2_weighted.bin')
```

```python
import galaxymidi

# Load the pre-computed comprehensive MIDI features
features = galaxymidi.load_features('Features/galaxy_midi_features.ldmb')
```

To audition the MIDIs, render them with any of the bundled SoundFonts using your favorite synthesizer (e.g., FluidSynth).

***

## Dataset structure

```
Galaxy MIDI Dataset
├── ATTRIBUTION.txt
├── Artwork
│   ├── Galaxy-MIDI-Dataset-Artwork (1).png
│   ├── Galaxy-MIDI-Dataset-Artwork (2).png
│   ├── Galaxy-MIDI-Dataset-Artwork (3).png
│   ├── Galaxy-MIDI-Dataset-Artwork (4).png
│   ├── Galaxy-MIDI-Dataset-Artwork (5).png
│   ├── Project-Los-Angeles.png
│   └── Tegridy-Code-2026.png
├── Code
│   ├── LICENSE
│   ├── midisimx-main.zip
│   └── tegridy-tools-main.zip
├── Embeddings
│   └── galaxy_midi_embeddings_1_2_1_2_weighted.bin
├── Features
│   └── galaxy_midi_features.ldmb
├── Files Lists
│   ├── aligned_midis_files_list.jsonl
│   ├── all_midis_files_list.jsonl
│   ├── clean_midis_files_list.jsonl
│   ├── drum_track_midis_files_list.jsonl
│   ├── genres_midis_files_list.jsonl
│   ├── karaoke_midis_files_list.jsonl
│   ├── mono_melodies_midis_files_list.jsonl
│   └── solo_melodies_midis_files_list.jsonl
├── GITHUB-REPO-LINK.txt
├── Genres
│   └── genres_data.jsonl
├── HUGGING-FACE-REPO-LINK.txt
├── Identified
│   └── identified_midis.jsonl
├── Karaoke
│   └── karaoke_midis_data.jsonl
├── LICENSE-CC-BY-NC-SA.txt
├── Lyrics
│   └── lyrics_midis.jsonl
├── MIDIs
│   └── ...
└── SoundFonts
    ├── Expressive Flute SSO-v1.2.sf2
    ├── KBH-Real-Choir-V2.5.sf2
    ├── Nice-Strings-PlusOrchestra-v1.6.sf2
    └── SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2

11 root directories, 19893571 MIDI files, 32 data files
```

***

## Dataset statistics

* 19893571 unique, standardized, and original MIDI files
* 17190286 pre-computed comprehensive MIDI features
* 15591559 pre-computed MIDI embeddings
* 9361365 identified MIDI files
* 2111480 MIDIs with monophonic melodies
* 754092 solo melody MIDI files
* 586398 solo drum track MIDI files
* 99475 Karaoke MIDI files
* 89569 MIDIs with genre, artist, and song labels
* 47069 MIDIs with corresponding English lyrics

***

## Dataset integrity verification info

* MD5: e23f34866b946ea56c989d24eb9ae48b
* SHA256: 1c0aade6ff554a268230ec0404b04ccbf91e386058db109a1845115c510eb9aa

***

## License

* The dataset is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** license (see `LICENSE-CC-BY-NC-SA.txt`). In short: you may share and adapt the dataset for **non-commercial** purposes, provided you give appropriate credit (see `ATTRIBUTION.txt` and [Citations](#citations)) and distribute derivatives under the same license.
* Bundled code in the `Code` directory is distributed under its own license terms — see `Code/LICENSE`.
* The bundled SoundFonts are subject to their respective original licenses.

***

## Citations

```bibtex
@misc{project_los_angeles_2026,
    author       = { Project Los Angeles and Tegridy Code },
    title        = { Galaxy MIDI Dataset },
    year         = 2026,
    url          = { https://huggingface.co/projectlosangeles/Galaxy-MIDI-Dataset },
    publisher    = { Hugging Face }
}
```

```bibtex
@misc{project_los_angeles_2025,
    author       = { Project Los Angeles },
    title        = { Discover-MIDI-Dataset },
    year         = 2025,
    url          = { https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset },
    publisher    = { Hugging Face }
}
```

```bibtex
@misc{TegridyMIDIDataset2025,
  title        = {Tegridy MIDI Dataset: Ultimate Multi-Instrumental MIDI Dataset for MIR and Music AI purposes},
  author       = {Alex Lev},
  publisher    = {Project Los Angeles / Tegridy Code},
  year         = {2025},
  url          = {https://github.com/asigalov61/Tegridy-MIDI-Dataset}
}
```

```bibtex
@misc {breadai_2025,
    author       = { {BreadAi} },
    title        = { Sourdough-midi-dataset (Revision cd19431) },
    year         = 2025,
    url          = {\url{https://huggingface.co/datasets/BreadAi/Sourdough-midi-dataset}},
    doi          = { 10.57967/hf/4743 },
    publisher    = { Hugging Face }
}
```

```bibtex
@inproceedings{bradshawaria,
  title={Aria-MIDI: A Dataset of Piano MIDI Files for Symbolic Music Modeling},
  author={Bradshaw, Louis and Colton, Simon},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=X5hrhgndxW}, 
}
```

***

### Project Los Angeles
### Tegridy Code 2026

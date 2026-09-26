# galaxymidi — API Reference

**galaxymidi** is a Python module for the **Galaxy MIDI Dataset**: dataset download,
parallel archive extraction, feature extraction, precomputed embeddings/features loading,
MIDI rendering, and file-list utilities. The companion `galaxymidi.helpers` module
provides MIDI normalization/fingerprinting and Linux package-management utilities.

| | |
|---|---|
| **Version** | 1.0.0 (`galaxymidi.__version__`) |
| **License** | Apache License 2.0 |
| **Source** | [github.com/Tegridy-Code/Project-Los-Angeles](https://github.com/Tegridy-Code/Project-Los-Angeles) |
| **Requirements** | `midisimx` (primary dependency), plus `huggingface_hub`, `midirenderer`, `tqdm`, and the package's sibling `TMIDIX` module |
| **Quick index** | [`API_FUNCTIONS_INDEX.md`](./API_FUNCTIONS_INDEX.md) |

## Installation

```bash
pip install midisimx
```

> **Import note:** both modules print a loading banner to stdout on import, and the
> main module sets the environment variable `HF_XET_HIGH_PERFORMANCE=1` for faster
> Hub transfers. The helpers module requires **Python 3.9+** (PEP 585 annotations).

## Quick Start

```python
import galaxymidi

galaxymidi.download_dataset()               # Fetch dataset archive from Hugging Face Hub
galaxymidi.parallel_extract()               # Extract archive to disk (parallel writes)

features   = galaxymidi.load_features()     # Precomputed feature store (LMBD)
embeddings = galaxymidi.load_embeddings()   # Precomputed embeddings (memory-mapped)
```

---

# Module: `galaxymidi`

<a name="download_dataset"></a>
## `download_dataset`

```python
galaxymidi.download_dataset(repo_id='projectlosangeles/Galaxy-MIDI-Dataset',
                            filename='Galaxy-MIDI-Dataset-CC-BY-NC-SA.tar.gz',
                            local_dir='./',
                            verbose=True,
                            **kwargs)
```

Download a dataset file from the Hugging Face Hub. A thin wrapper around
`huggingface_hub.hf_hub_download` (with `repo_type='dataset'`). If the file was
previously fetched, the cached path is returned instead of re-downloading.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `repo_id` | `str` | `'projectlosangeles/Galaxy-MIDI-Dataset'` | HF dataset repository (`'namespace/repo_name'`). |
| `filename` | `str` | `'Galaxy-MIDI-Dataset-CC-BY-NC-SA.tar.gz'` | Name of the file to download. |
| `local_dir` | `str` | `'./'` | Local destination directory (created if needed). |
| `verbose` | `bool` | `True` | Print progress messages. |
| `**kwargs` | | | Additional arguments forwarded to `hf_hub_download`. |

**Returns** — `str`: local path to the downloaded (or cached) file.

**Raises** — Propagates `huggingface_hub` exceptions (network/HTTP errors, missing
repo/file) and standard I/O errors.

**Note** — For private repositories, configure HF credentials in the environment beforehand.

---

<a name="parallel_extract"></a>
## `parallel_extract`

```python
galaxymidi.parallel_extract(tar_path='./Galaxy-MIDI-Dataset-CC-BY-NC-SA.tar.gz',
                            extract_path='./',
                            max_workers=64,
                            batch_size=8192)
```

Stream-extract a `.tar.gz` archive with concurrent disk writes. Members are read
sequentially from the archive stream (memory-efficient `r|gz` mode), read into memory
briefly, and written via a `ThreadPoolExecutor`. Scheduled writes are flushed and
awaited in batches to limit memory growth. Files already present at the target path
are **skipped**, making the function safe to re-run for resumption.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tar_path` | `str` | `'./Galaxy-MIDI-Dataset-CC-BY-NC-SA.tar.gz'` | Path to the archive. |
| `extract_path` | `str` | `'./'` | Destination directory (created if missing). |
| `max_workers` | `int` | `64` | Worker threads for concurrent writes. Tune for your storage device. |
| `batch_size` | `int` | `8192` | Futures buffered before flushing. Reduce for low-RAM environments. |

**Returns** — `None` (all writes are completed before return).

**Raises** — `FileNotFoundError` (missing archive), `tarfile.ReadError` (corrupt
archive), `OSError`/I/O errors; exceptions raised inside worker threads propagate
via `future.result()`.

**Notes**
- Streaming mode means no random access to archive members.
- The hard-coded `tqdm` progress bar assumes ~5,439,450 members (`total=5439450`).

---

<a name="load_features"></a>
## `load_features`

```python
galaxymidi.load_features(features_ldmb_file_path='./Galaxy-MIDI-Dataset/Features/galaxy_midi_features.ldmb')
```

Load the dataset's precomputed features file (LMBD format) via
`midisimx.ldmb.open_ldmb`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `features_ldmb_file_path` | `str` | `'./Galaxy-MIDI-Dataset/Features/galaxy_midi_features.ldmb'` | Path to the features `.ldmb` file. |

**Returns** — The features collection as returned by `midisimx.ldmb.open_ldmb`.

---

<a name="load_embeddings"></a>
## `load_embeddings`

```python
galaxymidi.load_embeddings(embeddings_bin_file_path='./Galaxy-MIDI-Dataset/Embeddings/galaxy_midi_embeddings_1_2_1_2_weighted.bin',
                           verbose=False)
```

Load the dataset's precomputed embeddings as a paired memory map via
`midisimx.memmap.load_paired_memmap`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `embeddings_bin_file_path` | `str` | `'./Galaxy-MIDI-Dataset/Embeddings/galaxy_midi_embeddings_1_2_1_2_weighted.bin'` | Path to the embeddings binary. |
| `verbose` | `bool` | `False` | Enable loader verbosity. |

**Returns** — The paired memory-mapped embeddings as returned by
`midisimx.memmap.load_paired_memmap`.

---

<a name="extract_midi_features"></a>
## `extract_midi_features`

```python
galaxymidi.extract_midi_features(input_midi)
```

Parse a MIDI file and compute a comprehensive feature dictionary. The internal
pipeline: parse to a single-track ms score → advanced score processing (sustain
applied, text/lyric events captured) → timing alignment stats → timing augmentation
(divided by 32) → cleanup (clean instruments, non-drum channel) → duplicate-pitch
removal → duration fixing → chordification with chord correction → monophonic
melody extraction → tokenization.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `input_midi` | `str` | Path to the input MIDI file. |

**Returns** — `dict` on success; **`None`** if parsing fails or the score contains
no notes. ⚠️ All exceptions are silently suppressed.

**Returned dictionary schema**

| Key | Type | Description |
|---|---|---|
| `midi_path` | `str` | Input MIDI path as passed. |
| *(analysis keys)* | various | Score analysis from `TMIDIX.advanced_score_processor`, with keys lowercased, spaces → `_`, and `number_of_` prefix stripped. The following raw keys are removed: `ticks_per_quarter_note`, `shortest_chord`, `longest_chord`, `score_patches`, `score_pitches`, `score_tones`, `bad_chords`. |
| `text_lyric_latin` | | Latin text/lyric event data (renamed from `all_text_and_lyric_events_latin`). |
| `aligned` | `dict` | Most common inter-note delta: `{'dtime_ms', 'aligned', 'total'}`. |
| `karaoke` | `dict` | `{'text': n, 'lyric': n}` — counts of text and lyric events. |
| `run_time` | `dict` | `{'total', 'last_time'}` — score run time (ms). |
| `clean_midi` | `dict` | `{'clean', 'total'}` — notes on clean instruments (non-drum channel) vs. all notes. |
| `dupe_pitches` | `dict` | `{'deduped', 'total'}` — note counts before/after duplicate-pitch removal. |
| `bad_durs` | `dict` | `{'bad', 'count', 'zero', 'total'}` — duration statistics (min duration 128). |
| `mono_mels` | `dict` | Monophonic melodies keyed by instrument/patch. |
| `features_counts` | `dict` | Token ID → count (see table below). |
| `pitches_patches_counts` | `dict` | `(pitch, patch)` tuple → count. |

**`features_counts` token layout** (vocabulary size 978, plus 2 meta tokens):

| Token range | Meaning |
|---|---|
| 0–127 | Delta time (ms) |
| 128–255 | Duration (`+128`) |
| 256–383 | Pitch (`+256`) |
| 384–511 | Velocity (`+384`) |
| 512–640 | Patch / program (`+512`) |
| 641–656 | Channel (`+641`) |
| 657–977 | Chord type (`+657`, index into `TMIDIX.ALL_CHORDS_SORTED`) |
| `978` | Count of auto-corrected ("bad") chords |
| `979` | Bar count (128 ms buckets) |

---

<a name="render_midi"></a>
## `render_midi`

```python
galaxymidi.render_midi(input_midi_file,
                       output_wav_file='',
                       sf2_path='./Galaxy-MIDI-Dataset/SoundFonts/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2',
                       verbose=True)
```

Render a MIDI file to WAV using a SoundFont via `midirenderer.render_wave_from`.
Both the SoundFont and MIDI are read fully into memory as bytes. If
`output_wav_file` is empty, the output path is auto-derived from the input
(`song.mid` → `song.wav`). Existing files are overwritten without prompting.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_midi_file` | `str` | — | Path to the input MIDI file. |
| `output_wav_file` | `str` | `''` | Output WAV path; auto-derived if empty. |
| `sf2_path` | `str` | Dataset SGM SoundFont path | Path to the SoundFont (`.sf2`) file. |
| `verbose` | `bool` | `True` | Print progress messages. |

**Returns** — `str`: path to the saved WAV file.

**Raises** — `FileNotFoundError` (missing input/SoundFont), I/O errors on write,
renderer exceptions propagate.

---

<a name="read_jsonl"></a>
## `read_jsonl`

```python
galaxymidi.read_jsonl(file_name='./Galaxy-MIDI-Dataset/Files Lists/all_midis_files_list',
                      file_ext='.jsonl',
                      verbose=True)
```

Read a JSONL file line by line into a list of parsed records. The extension
`file_ext` is appended if the given path has no extension. Corrupted JSON lines are
skipped with a warning. On `KeyboardInterrupt`, reading stops gracefully and all
records collected so far are returned.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file_name` | `str` | `'./Galaxy-MIDI-Dataset/Files Lists/all_midis_files_list'` | Path to the file. |
| `file_ext` | `str` | `'.jsonl'` | Extension appended when the path lacks one. |
| `verbose` | `bool` | `True` | Print progress and error messages. |

**Returns** — `list[dict]`: parsed JSON records.

---

<a name="write_file"></a>
## `write_file` *(internal helper)*

```python
galaxymidi.write_file(data: bytes, target_path: str)
```

Create the parent directory of `target_path` (if any) and write `data` to it in
binary mode. Used by `parallel_extract` as the concurrent write worker.

**Returns** — `None`.

---

# Module: `galaxymidi.helpers`

Utility helpers for the galaxymidi package: MIDI normalization and fingerprinting,
aligned-list sorting, and Linux system-package management.

| | |
|---|---|
| **Version** | 1.0.0 (`galaxymidi.helpers.__version__`) |
| **Dependencies** | Python standard library only, plus the package's internal `TMIDIX` module — no extra installs required |
| **Platform** | MIDI helpers are cross-platform; `is_installed` / `install_apt_package` are **Debian/Ubuntu (Linux) only** |

## Usage Example

```python
from galaxymidi import helpers

# Fingerprint a MIDI file (signature/metadata-independent)
info = helpers.get_normalized_midi_md5_hash('song.mid')
print(info['normalized_md5'])

# Canonicalize a MIDI file and save it
out_path = helpers.normalize_midi_file('song.mid', output_dir='./normalized')

# Sort dataset lists by sequence length (in place, alignment preserved)
helpers.sort_aligned_lists(midi_names, midi_sequences, reverse=True)

# System package management (Linux/Debian only)
helpers.install_apt_package('fluidsynth')
```

---

<a name="sort_aligned_lists"></a>
## `sort_aligned_lists`

```python
galaxymidi.helpers.sort_aligned_lists(midi_names, midi_sequences, reverse=False)
```

Sort two aligned lists **in place** based on the length of each MIDI sequence. The
element index positions are stably sorted by `len(midi_sequences[i])`, and both
lists are rewritten via slice assignment so they remain aligned afterward.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `midi_names` | `list[str]` | — | MIDI file names; must be aligned with `midi_sequences`. |
| `midi_sequences` | `list[list[int]]` | — | MIDI sequences (or any list-likes); each element's length is the sort key. |
| `reverse` | `bool` | `False` | `True` for descending (longest first); `False` for ascending (shortest first). |

**Returns** — `None`. Both input lists are mutated in place and remain aligned.

**Notes** — The sort is **stable** (Python's `sorted()`), and fully **in-place** —
no new lists are returned or bound.

---

<a name="get_normalized_midi_md5_hash"></a>
## `get_normalized_midi_md5_hash`

```python
galaxymidi.helpers.get_normalized_midi_md5_hash(midi_file)
```

Compute both the original and a **normalized MD5 hash** for a MIDI file. The file
bytes are parsed with `TMIDIX.midi2score` (MIDI-signature check disabled) and
re-serialized with `TMIDIX.score2midi`; the MD5 of the re-serialized bytes serves as
a canonical fingerprint. Useful for detecting duplicate MIDIs regardless of
header/metadata differences.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `midi_file` | `str` | Path to the input MIDI file. |

**Returns** — `dict`:

| Key | Type | Description |
|---|---|---|
| `'midi_name'` | `str` | File basename without extension. |
| `'original_md5'` | `str` | MD5 hex digest of the raw file bytes. |
| `'normalized_md5'` | `str` | MD5 hex digest of the normalized (re-serialized) MIDI. |

**Raises** — `FileNotFoundError` if the file is missing; `TMIDIX` parse errors propagate.

**Note** — The entire file is read into memory (fine for typical MIDI sizes).

---

<a name="normalize_midi_file"></a>
## `normalize_midi_file`

```python
galaxymidi.helpers.normalize_midi_file(midi_file, output_dir='', output_file_name='')
```

Normalize a MIDI file — parse with `TMIDIX.midi2score` (signature check disabled)
and re-serialize with `TMIDIX.score2midi` — and write the canonical result to disk.
Same normalization pipeline as
[`get_normalized_midi_md5_hash`](#get_normalized_midi_md5_hash).

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `midi_file` | `str` | — | Path to the input MIDI file. |
| `output_dir` | `str` | `''` | Destination directory; defaults to the current working directory. Created if missing. |
| `output_file_name` | `str` | `''` | Output file name; defaults to the input file's basename. |

**Returns** — `str`: path to the written normalized MIDI file.

**Behavior** — If the target path already exists, the output is written as
`<stem>_normalized.mid` instead — **existing files are never overwritten**.

**Raises** — `FileNotFoundError` (missing input), `OSError` (write errors);
`TMIDIX` parse errors propagate.

---

<a name="is_installed"></a>
## `is_installed`

```python
galaxymidi.helpers.is_installed(pkg)
```

Check whether a Debian/Ubuntu package is installed, using
`dpkg-query -W -f=${Status}`. Returns `True` if the status output contains
`installed`.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `pkg` | `str` | Debian package name (e.g., `'fluidsynth'`). |

**Returns** — `bool`: `True` if installed, `False` if unknown to dpkg
(`CalledProcessError` is handled).

**Raises** — `FileNotFoundError` on systems without `dpkg` installed (only
`CalledProcessError` is caught).

**Notes** — Debian/Ubuntu only; `dpkg-query` is invoked twice per call (once with
`check=True`, once to read the status string).

---

<a name="install_apt_package"></a>
## `install_apt_package`

```python
galaxymidi.helpers.install_apt_package(pkg,
                                       update=True,
                                       timeout=600,
                                       require_root=True,
                                       use_python_apt=False)
```

Install an apt package **idempotently**, with retries and verification.

**Workflow**

1. If `pkg` is already installed → returns immediately with `'already_installed'`.
2. If `use_python_apt=True` → attempts the python-apt API first (requires
   `python-apt` and root); on success returns `'installed_via_python_apt'`. Any
   exception silently falls through to the `apt-get` path.
3. **Privilege check:** if not running as root and `require_root=True`, commands are
   prefixed with `sudo`; raises `PermissionError` if `sudo` is unavailable.
4. **Optional update:** `apt-get update` with up to **5 attempts** and exponential
   backoff (`2^attempt` seconds).
5. **Install:** `apt-get -y install <pkg>` with up to **6 attempts**; retries with
   backoff when dpkg lock (`"Could not get lock"`) or interruption
   (`"dpkg was interrupted"`) errors are detected; other failures raise immediately.
   After a successful install, the result is verified with `is_installed`.
6. Raises `RuntimeError` if the package still isn't installed after all retries.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pkg` | `str` | — | Package name (e.g., `'fluidsynth'`). |
| `update` | `bool` | `True` | Run `apt-get update` first. |
| `timeout` | `int` | `600` | Per-command timeout in seconds (applies to each `apt-get` invocation). |
| `require_root` | `bool` | `True` | Prefix with `sudo` when not running as root. |
| `use_python_apt` | `bool` | `False` | Try the python-apt API first. |

**Returns** — `dict`:

| Key | Values |
|---|---|
| `'status'` | `'already_installed'` \| `'installed_via_python_apt'` \| `'installed'` |
| `'package'` | The package name as passed. |

**Raises**

- `PermissionError` — root privileges required and `sudo` unavailable.
- `RuntimeError` — installation failed after retries, or `apt-get` succeeded but
  verification failed.
- `subprocess.CalledProcessError` — `apt-get update`/`install` failed permanently.
- `subprocess.TimeoutExpired` — an apt operation exceeded `timeout`.

**Notes** — Debian/Ubuntu only; apt output is captured (not streamed to the
console); typically requires root or passwordless `sudo`.

---

<a name="_run_apt_get"></a>
## `_run_apt_get` *(internal helper)*

```python
galaxymidi.helpers._run_apt_get(args, timeout)
```

Run an `apt-get` command with forced-configuration options:

```
apt-get -y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold <args...>
```

Executed via `subprocess.run` with `check=True` and captured output.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `args` | `list[str]` | `apt-get` subcommand and arguments (e.g., `['install', 'fluidsynth']`). |
| `timeout` | `int` | Timeout in seconds. |

**Returns** — `subprocess.CompletedProcess`.

**Raises** — `subprocess.CalledProcessError`, `subprocess.TimeoutExpired`.

**Note** — Currently **not called** by other functions in the module
(`install_apt_package` constructs its own commands); kept for internal/future use.

---

## Module Attributes

| Attribute | Value |
|---|---|
| `galaxymidi.__version__` | `'1.0.0'` |
| `galaxymidi.helpers.__version__` | `'1.0.0'` |

---

### Project Los Angeles
### Tegridy Code 2026
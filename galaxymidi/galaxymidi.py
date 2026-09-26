r'''###############################################################################
###################################################################################
#
#
#	galaxymidi Python Module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2026
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
#
###################################################################################
###################################################################################
#
#   Copyright 2026 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
###################################################################################
#
#   Module dependencies
#
#   !pip install midisimx
#
###################################################################################
###################################################################################
#
#   Basic use example
#
#   import galaxymidi
#
#   galaxymidi.download_dataset()
#
#   galaxymidi.parallel_extract()
#
#   embeddings = galaxymidi.load_embeddings()
#
#   features = galaxymidi.load_features()
#
###################################################################################
'''

###################################################################################
###################################################################################

print('=' * 70)
print('Loading galaxymidi Python module...')
print('Please wait...')

__version__ = '1.0.0'

###################################################################################
###################################################################################

import os

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from collections import Counter

import json

from midisimx import ldmb, memmap

from . import TMIDIX

import midirenderer

from pathlib import Path

import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download

import tqdm

###################################################################################

def download_dataset(repo_id='projectlosangeles/Galaxy-MIDI-Dataset',
                     filename='Galaxy-MIDI-Dataset-CC-BY-NC-SA.tar.gz',
                     local_dir='./',
                     verbose=True,
                     **kwargs
                    ):

    """Download the Discover MIDI Dataset archive from the Hugging Face Hub.
    
    This helper wraps `huggingface_hub.hf_hub_download` to fetch a dataset file
    from a specified repository on the Hugging Face Hub and save it to a local
    directory. It returns the absolute path to the downloaded file (or the
    cached file path if the file was already present in the local cache).
    
    Parameters
    ----------
    repo_id : str, optional
        Identifier of the Hugging Face dataset repository in the form
        `'namespace/repo_name'`. Default: `'projectlosangeles/Galaxy-MIDI-Dataset'`.
    filename : str, optional
        Name of the file to download from the repository (for example, a tarball
        or zip archive). Default:
        `'Galaxy-MIDI-Dataset-CC-BY-NC-SA.tar.gz'`.
    local_dir : str, optional
        Local directory where the downloaded file will be stored or cached.
        The directory will be created by the underlying `hf_hub_download` if
        necessary. Default: `'./'`.
    kwargs : dict, optional
        Additional args for hf_hub_download function
    
    Returns
    -------
    str
        Absolute path to the downloaded file on the local filesystem. If the
        file already exists in the Hugging Face cache, the cached path is
        returned.
    
    Raises
    ------
    Exception
        Propagates exceptions raised by `huggingface_hub.hf_hub_download` (for
        example network/HTTP errors, repository or filename not found) and
        standard I/O errors (e.g., `OSError`) that may occur while writing to
        disk.
    
    Notes
    -----
    - This function relies on the `huggingface_hub` package and its
      authentication/caching behavior. If the repository is private, ensure
      that the environment is configured with appropriate HF credentials.
    - `hf_hub_download` may return a cached path instead of re-downloading the
      file if the same file has been previously fetched.
    - Use this helper when you want a simple, single-call way to obtain the
      dataset archive and receive its local path for subsequent processing.

    """
    
    if verbose:
        print('=' * 70)
        print('Downloading dataset...')
        print('=' * 70)

    result = hf_hub_download(repo_id=repo_id,
                             repo_type='dataset',
                             filename=filename,
                             local_dir=local_dir,
                             **kwargs
                            )
    
    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)
    
    return result

###################################################################################

def write_file(data: bytes, target_path: str):

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with open(target_path, "wb") as f:
        f.write(data)

###################################################################################

def parallel_extract(tar_path: str = './Galaxy-MIDI-Dataset-CC-BY-NC-SA.tar.gz',
                     extract_path: str = './', 
                     max_workers: int = 64, 
                     batch_size: int = 8192
                    ):

    """Extract a large tar.gz archive to disk using parallel writes.
    
    This helper streams a tar.gz archive and extracts its members to `extract_path`
    while performing file writes concurrently with a thread pool. It is optimized
    for very large archives by avoiding loading the entire archive into memory and
    by batching completed write futures to limit memory growth.
    
    Behavior summary
    - Opens the archive in streaming mode (`tarfile.open(..., mode="r|gz")`) so
      members are read sequentially from the archive stream.
    - Creates directories on demand and tracks created directories in `created_dirs`
      to avoid redundant `os.makedirs` calls.
    - Skips extraction for files that already exist at the target path.
    - Reads each file member into memory briefly, then schedules `write_file(data, path)`
      on a `ThreadPoolExecutor` for concurrent disk writes.
    - Flushes and waits for a batch of futures once `batch_size` futures are queued,
      and again at the end of processing to ensure all writes complete.
    
    Parameters
    ----------
    tar_path : str, optional
        Path to the tar.gz archive to extract. Default:
        `'./Galaxy-MIDI-Dataset-CC-BY-NC-SA.tar.gz'`.
    extract_path : str, optional
        Destination directory where archive members will be extracted. The directory
        is created if it does not exist. Default: `'./'`.
    max_workers : int, optional
        Maximum number of worker threads used for concurrent file writes. Higher
        values increase parallelism but also increase contention and resource use.
        Choose a value appropriate for your system and storage device. Default: 64.
    batch_size : int, optional
        Number of scheduled write futures to accumulate before waiting for them to
        complete. This limits the number of in-flight futures and the memory used
        to buffer file contents. Default: 8192.
    
    Returns
    -------
    None
        The function performs extraction as a side effect and does not return a
        value. All scheduled write tasks are awaited before the function returns.
    
    Dependencies and requirements
    - Requires `tarfile`, `os`, `tqdm`, `concurrent.futures.ThreadPoolExecutor`,
      and a callable `write_file(data: bytes, path: str)` available in scope.
    - `write_file` must be thread-safe and handle creating parent directories if
      necessary (the function already creates directories for `member.isdir()` but
      `write_file` should be robust).
    
    Exceptions
    ----------
    FileNotFoundError
        If `tar_path` does not exist.
    tarfile.ReadError
        If the archive cannot be read as a tar file or is corrupted.
    OSError, IOError, PermissionError
        For I/O errors while reading the archive or writing files to disk.
    RuntimeError
        If a worker thread raises an exception during `write_file`, that exception
        will be propagated when awaiting the future.
    
    Notes and recommendations
    - The archive is opened in streaming mode (`"r|gz"`). This is memory-efficient
      but means random access to archive members is not possible.
    - `batch_size` controls memory usage: each scheduled future holds the file's
      bytes in memory until the future is awaited. Reduce `batch_size` for low-RAM
      environments.
    - `max_workers` should be tuned for the target storage medium. Very large
      values can cause contention and degrade throughput on HDDs or network filesystems.
    - The function uses a hard-coded `tqdm` progress bar with `total=5439450` and
      `miniters=100`. If your archive has a different number of members, consider
      adjusting or removing the `total` argument to avoid misleading progress.
    - The function skips extraction when a target file already exists. If you need
      to overwrite existing files, remove the existence check or add an overwrite flag.
    
    """
    
    os.makedirs(extract_path, exist_ok=True)
    
    created_dirs = set()
    futures = []

    with tarfile.open(tar_path, mode="r|gz") as tar, \
         ThreadPoolExecutor(max_workers=max_workers) as executor:

        for member in tqdm.tqdm(tar, total=5439450, miniters=100):
            
            target_path = os.path.join(extract_path, member.name)
            
            if member.isdir():
                if target_path not in created_dirs:
                    os.makedirs(target_path, exist_ok=True)
                    created_dirs.add(target_path)
                    
            elif member.isfile():

                if os.path.exists(target_path):
                    continue

                fobj = tar.extractfile(member)
                
                if fobj is None:
                    continue
                    
                data = fobj.read()
                futures.append(executor.submit(write_file, data, target_path))

                if len(futures) >= batch_size:
                    for future in as_completed(futures):
                        future.result()
                    futures = []
                    
        for future in as_completed(futures):
            future.result()

###################################################################################

def load_features(features_ldmb_file_path='./Galaxy-MIDI-Dataset/Features/galaxy_midi_features.ldmb'):
    return ldmb.open_ldmb(features_ldmb_file_path)
    

###################################################################################

def load_embeddings(embeddings_bin_file_path='./Galaxy-MIDI-Dataset/Embeddings/galaxy_midi_embeddings_1_2_1_2_weighted.bin',
                    verbose=False
                    ):
    return memmap.load_paired_memmap(embeddings_bin_file_path, verbose=verbose)

###################################################################################

def extract_midi_features(input_midi):

    try:
        raw_score = TMIDIX.midi2single_track_ms_score(input_midi, do_not_check_MIDI_signature=True)
        
        data = TMIDIX.advanced_score_processor(raw_score,
                                               return_score_analysis=True,
                                               return_enhanced_score_notes=True,
                                               return_text_and_lyric_events=True,
                                               apply_sustain=True
                                              )
    
        if len(data) == 3:
            analysis, raw_escore_notes, text_events = data
    
        else:
            analysis, raw_escore_notes = data
            text_events = []
            
        if raw_escore_notes:
    
            #======================================================================================
    
            dscore = TMIDIX.delta_score_notes(raw_escore_notes, timings_clip_value=3999)
            dtimes = [e[1] for e in dscore if e[1] != 0]
            mc_dtime_count = Counter(dtimes).most_common(1)[0]
    
            aligned = {'dtime_ms': mc_dtime_count[0],
                       'aligned': mc_dtime_count[1],
                       'total': len(dtimes)
                      }
    
            #======================================================================================
    
            text_ev = [e for e in text_events if e[0] == 'text_event']
            lyric_ev = [e for e in text_events if e[0] == 'lyric']
    
            karaoke = {'text': len(text_ev),
                       'lyric': len(lyric_ev)
                      }
    
            #======================================================================================
    
            run_time = TMIDIX.escore_notes_run_time(raw_escore_notes)
    
            run_time = {'total': run_time[0], 'last_time': run_time[1]}
    
            #======================================================================================
    
            escore_notes = TMIDIX.augment_enhanced_score_notes(raw_escore_notes, timings_divider=32)
    
            #======================================================================================
    
            clean_escore_notes = [e for e in escore_notes if e[6] in TMIDIX.CLEAN_INSTRUMENTS and e[3] != 9]
    
            clean_midi = {'clean': len(clean_escore_notes), 'total': len(escore_notes)}
    
            #======================================================================================
    
            dd_escore_notes = TMIDIX.remove_duplicate_pitches_from_escore_notes(escore_notes)
    
            dupe_pitches = {'deduped': len(dd_escore_notes), 'total': len(escore_notes)}
    
            #======================================================================================
    
            bad_durs_stats = TMIDIX.escore_notes_durations_counter(dd_escore_notes, min_duration=128)
    
            bad_durs_stats = {'bad': bad_durs_stats[0],
                              'count': bad_durs_stats[3],
                              'zero': bad_durs_stats[2],
                              'total': bad_durs_stats[1],
                             }
    
            fixed_escore_notes = TMIDIX.fix_escore_notes_durations(dd_escore_notes, min_notes_gap=0)
    
            #======================================================================================
            
            cscore = TMIDIX.chordify_score([1000, fixed_escore_notes])
    
            fixed_score = []
    
            bad_chords_counter = 0
    
            for c in cscore:
    
                tones_chord = sorted(set([e[4] % 12 for e in c if e[3] != 9]))
    
                if tones_chord:
                    if tones_chord not in TMIDIX.ALL_CHORDS_SORTED:
                        tones_chord = TMIDIX.check_and_fix_tones_chord(tones_chord, use_full_chords=False)
    
                        bad_chords_counter += 1
        
                for e in c:
                    if e[4] % 12 in tones_chord or e[3] == 9:
                        fixed_score.append(e)
    
            #======================================================================================
    
            mono_mels = TMIDIX.escore_notes_monoponic_melodies([e for e in fixed_score if e[3] != 9])
    
            mono_mels = {k: v for k, v in mono_mels}
    
            #======================================================================================
    
            cscore = TMIDIX.chordify_score([1000, fixed_score])
    
            score = []

            pp_counter = Counter()
    
            abs_time = 0
            pbar = -1
            bars_count = 0
    
            pc = cscore[0]
    
            for c in cscore:
                
                if abs_time // 128 > pbar:
                    bars_count += 1
                    pbar = abs_time // 128
    
                tones_chord = sorted(set([e[4] % 12 for e in c if e[3] != 9]))
    
                if tones_chord:
    
                    if len(c) > 1:
                        chord_tok = TMIDIX.ALL_CHORDS_SORTED.index(tones_chord)
                        score.append(chord_tok+657) # Total vocab size 978
    
                dtime = max(0, min(127, c[0][1]-pc[0][1]))
    
                if dtime != 0:
                    score.append(dtime)
    
                abs_time += dtime
    
                for e in c:
                    score.extend([max(1, min(127, e[2]))+128, # Durs
                                  max(1, min(127, e[4]))+256, # Ptcs
                                  max(1, min(127, e[5]))+384, # Vels
                                  max(0, min(128, e[6]))+512, # Pats
                                  max(0, min(15, e[3]))+641   # Chans
                                  # Total 657
                                 ])

                    pp_counter[(max(1, min(127, e[4])), max(0, min(128, e[6])))] += 1
    
                pc = c
    
            #======================================================================================
    
            features_counter = Counter(score)
            features_counter[978] = bad_chords_counter
            features_counter[979] = bars_count
    
            #======================================================================================
    
            final_dict = {'midi_path': input_midi}
            
            final_dict |= {k.lower().replace(' ', '_').replace('number_of_', ''): v for k, v in analysis}
    
            del final_dict['ticks_per_quarter_note']
            del final_dict['shortest_chord']
            del final_dict['longest_chord']
            del final_dict['score_patches']
            del final_dict['score_pitches']
            del final_dict['score_tones']
            del final_dict['bad_chords']
            
            final_dict['text_lyric_latin'] = final_dict.pop('all_text_and_lyric_events_latin', None)
    
            final_dict['aligned'] = aligned
            final_dict['karaoke'] = karaoke
            final_dict['run_time'] = run_time
            final_dict['clean_midi'] = clean_midi
            final_dict['dupe_pitches'] = dupe_pitches
            final_dict['bad_durs'] = bad_durs_stats
            final_dict['mono_mels'] = mono_mels
            final_dict['features_counts'] = {k: v for k, v in features_counter.most_common()}
            final_dict['pitches_patches_counts'] = {k: v for k, v in pp_counter.most_common()}
    
            return final_dict

    except:
        pass
    
###################################################################################

def render_midi(input_midi_file,
                output_wav_file='',
                sf2_path='./Galaxy-MIDI-Dataset/SoundFonts/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2',
                verbose=True
               ):
    
    """Render a MIDI file to a WAV file using a SoundFont and a MIDI renderer.
    
    This helper reads a SoundFont (`.sf2`) and a MIDI file, uses the `midirenderer`
    interface to synthesize audio, and writes the resulting waveform to disk.
    It is a thin wrapper around `midirenderer.render_wave_from` that handles
    file I/O, default output naming, and optional progress printing.
    
    Parameters
    ----------
    input_midi_file : str or pathlib.Path
        Path to the input MIDI file to render. The file will be read as bytes
        and passed to the renderer.
    output_wav_file : str, optional
        Path where the rendered WAV will be saved. If empty or not provided,
        the function will create a file next to `input_midi_file` with the same
        base name and a `.wav` extension (for example, `song.mid` -> `song.wav`).
        Default: '' (auto-derived from `input_midi_file`).
    sf2_path : str or pathlib.Path, optional
        Path to the SoundFont (`.sf2`) file used for synthesis. The file is
        read as bytes and supplied to the renderer. Default:
        `'./Galaxy-MIDI-Dataset/SoundFonts/SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2'`.
    verbose : bool, optional
        If True, print simple progress messages to stdout. Default: True.
    
    Returns
    -------
    str
        The path to the saved WAV file (the same value as `output_wav_file` after
        any default naming logic is applied).
    
    Raises
    ------
    FileNotFoundError
        If `input_midi_file` or `sf2_path` does not exist when attempting to read.
    OSError, IOError, PermissionError
        If there is an error reading the input files or writing the output WAV.
    ValueError, RuntimeError
        If `midirenderer.render_wave_from` fails or returns invalid data.
    TypeError
        If the renderer returns a non-bytes object or if provided arguments are
        of incompatible types.
    
    Notes
    -----
    - This function expects a `midirenderer` object in scope that exposes the
      method `render_wave_from(sf2_bytes: bytes, midi_bytes: bytes) -> bytes`.
      The renderer must accept raw bytes for the SoundFont and MIDI and return
      raw WAV bytes suitable for writing directly to a `.wav` file.
    - The function reads the entire SoundFont and MIDI into memory before
      rendering. For very large SoundFonts or constrained environments, ensure
      sufficient memory is available.
    - The function writes the returned bytes directly to disk without additional
      WAV header manipulation; the renderer is expected to return a complete,
      valid WAV byte stream.
    - If you need to overwrite existing files, ensure `output_wav_file` points to
      the desired path; this function will overwrite without prompting.
    
    """
    if verbose:
        print('=' * 70)
        print('Rendering MIDI...')
        
    wav_data = midirenderer.render_wave_from(
        Path(sf2_path).read_bytes(),
        Path(input_midi_file).read_bytes()
    )

    if verbose:
        print('Done!')
        print('=' * 70)
    
        print('Saving rendered MIDI...')

    if not output_wav_file:
        output_wav_file = os.path.splitext(input_midi_file)[0] + '.wav'
    
    with open(output_wav_file, 'wb') as fi:
        fi.write(wav_data)

    if verbose:
        print('Done!')
        print('=' * 70)

    return output_wav_file

###################################################################################

def read_jsonl(file_name='./Galaxy-MIDI-Dataset/Files Lists/all_midis_files_list', 
               file_ext='.jsonl', 
               verbose=True
              ):

    if verbose:
        print('=' * 70)
        print('Reading jsonl file...')
        print('=' * 70)

    if not os.path.splitext(file_name)[1]:
        file_name += file_ext

    with open(file_name, 'r', encoding='utf-8') as f:

        records = []
        gl_count = 0
        
        for i, line in tqdm.tqdm(enumerate(f), disable=not verbose):
            
            try:
                record = json.loads(line)
                records.append(record)
                gl_count += 1

            except KeyboardInterrupt:
                if verbose:
                    print('=' * 70)
                    print('Stoping...')
                    print('=' * 70)
                    
                f.close()
    
                return records
               
            except json.JSONDecodeError:
                if verbose:
                    print('=' * 70)
                    print('[ERROR] Line', i, 'is corrupted! Skipping it...')
                    print('=' * 70)
                    
                continue
                
    f.close()
    
    if verbose:
        print('=' * 70)
        print('Loaded total of', gl_count, 'jsonl records.')
        print('=' * 70)
        print('Done!')
        print('=' * 70)

    return records

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the galaxymidi Python module
###################################################################################
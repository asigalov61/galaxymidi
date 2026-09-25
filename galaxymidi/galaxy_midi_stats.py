#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  GALAXY MIDI DATASET — FEATURE-STATISTICS COLLECTOR, REPORTER & VISUALIZER
#
#  Branding   : Galaxy MIDI Dataset
#  Attribution: Project Los Angeles --- Tegridy Code 2026
#
#  PROGRAMMATIC USE (recommended for very large datasets, e.g. 10M+ dicts):
#
#      from galaxy_midi_stats import Config, analyze_features
#      cfg = Config(verbosity=1, workers=32, backend='loky', batch_size=20000)
#      stats, paths = analyze_features(my_list_of_feature_dicts, cfg)
#      # Items may be bare feature dicts OR {md5_hash: feature_dict} wrappers
#      # (both auto-detected; hashes are preserved as file IDs).
#      # Produces the same outputs as CLI: report.txt, stats_summary.json, plots/.
#
#  CLI (JSON files/dirs of feature dicts, and/or raw MIDIs if TMIDIX installed):
#      python galaxy_midi_stats.py --demo -v 2
#      python galaxy_midi_stats.py ./features -j 8 -v 2
#
#  Deps: numpy + matplotlib (plots); joblib / tqdm (optional).
# ============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import traceback
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except Exception:
    HAVE_JOBLIB = False

try:
    from tqdm.auto import tqdm          # notebook-friendly progress bars
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

try:
    import TMIDIX                        # optional: chord decode + MIDI extraction
    HAVE_TMIDIX = True
except Exception:
    TMIDIX = None
    HAVE_TMIDIX = False

# ------------------------------------------------------------------ branding
DEFAULT_BRANDING = 'Galaxy MIDI Dataset'
DEFAULT_ATTRIBUTION = 'Project Los Angeles --- Tegridy Code 2026'
REPORT_WIDTH = 100

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

GM_PATCHES = [
    'Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Grand Piano', 'Honky-tonk Piano',
    'Electric Piano 1', 'Electric Piano 2', 'Harpsichord', 'Clavi', 'Celesta', 'Glockenspiel',
    'Music Box', 'Vibraphone', 'Marimba', 'Xylophone', 'Tubular Bells', 'Dulcimer',
    'Drawbar Organ', 'Percussive Organ', 'Rock Organ', 'Church Organ', 'Reed Organ', 'Accordion',
    'Harmonica', 'Tango Accordion', 'Acoustic Guitar (nylon)', 'Acoustic Guitar (steel)',
    'Electric Guitar (jazz)', 'Electric Guitar (clean)', 'Electric Guitar (muted)',
    'Overdriven Guitar', 'Distortion Guitar', 'Guitar Harmonics', 'Acoustic Bass',
    'Electric Bass (finger)', 'Electric Bass (pick)', 'Fretless Bass', 'Slap Bass 1', 'Slap Bass 2',
    'Synth Bass 1', 'Synth Bass 2', 'Violin', 'Viola', 'Cello', 'Contrabass', 'Tremolo Strings',
    'Pizzicato Strings', 'Orchestral Harp', 'Timpani', 'String Ensemble 1', 'String Ensemble 2',
    'Synth Strings 1', 'Synth Strings 2', 'Choir Aahs', 'Voice Oohs', 'Synth Voice',
    'Orchestra Hit', 'Trumpet', 'Trombone', 'Tuba', 'Muted Trumpet', 'French Horn',
    'Brass Section', 'Synth Brass 1', 'Synth Brass 2', 'Soprano Sax', 'Alto Sax', 'Tenor Sax',
    'Baritone Sax', 'Oboe', 'English Horn', 'Bassoon', 'Clarinet', 'Piccolo', 'Flute',
    'Recorder', 'Pan Flute', 'Blown Bottle', 'Shakuhachi', 'Whistle', 'Ocarina',
    'Lead 1 (square)', 'Lead 2 (sawtooth)', 'Lead 3 (calliope)', 'Lead 4 (chiff)',
    'Lead 5 (charang)', 'Lead 6 (voice)', 'Lead 7 (fifths)', 'Lead 8 (bass + lead)',
    'Pad 1 (new age)', 'Pad 2 (warm)', 'Pad 3 (polysynth)', 'Pad 4 (choir)', 'Pad 5 (bowed)',
    'Pad 6 (metallic)', 'Pad 7 (halo)', 'Pad 8 (sweep)', 'FX 1 (rain)', 'FX 2 (soundtrack)',
    'FX 3 (crystal)', 'FX 4 (atmosphere)', 'FX 5 (brightness)', 'FX 6 (goblins)', 'FX 7 (echoes)',
    'FX 8 (sci-fi)', 'Sitar', 'Banjo', 'Shamisen', 'Koto', 'Kalimba', 'Bag Pipe', 'Fiddle',
    'Shanai', 'Tinkle Bell', 'Agogo', 'Steel Drums', 'Woodblock', 'Taiko Drum', 'Melodic Tom',
    'Synth Drum', 'Reverse Cymbal', 'Guitar Fret Noise', 'Breath Noise', 'Seashore', 'Bird Tweet',
    'Telephone Ring', 'Helicopter', 'Applause', 'Gunshot', 'Drums',
]

# Standard GM Level-1 percussion note map (drum channel / patch 128).
GM_DRUMS = {
    35: 'Acoustic Bass Drum', 36: 'Bass Drum 1', 37: 'Side Stick', 38: 'Acoustic Snare',
    39: 'Hand Clap', 40: 'Electric Snare', 41: 'Low Floor Tom', 42: 'Closed Hi-Hat',
    43: 'High Floor Tom', 44: 'Pedal Hi-Hat', 45: 'Low Tom', 46: 'Open Hi-Hat',
    47: 'Low-Mid Tom', 48: 'Hi-Mid Tom', 49: 'Crash Cymbal 1', 50: 'High Tom',
    51: 'Ride Cymbal 1', 52: 'Chinese Cymbal', 53: 'Ride Bell', 54: 'Tambourine',
    55: 'Splash Cymbal', 56: 'Cowbell', 57: 'Crash Cymbal 2', 58: 'Vibraslap',
    59: 'Ride Cymbal 2', 60: 'Hi Bongo', 61: 'Low Bongo', 62: 'Mute Hi Conga',
    63: 'Open Hi Conga', 64: 'Low Conga', 65: 'High Timbale', 66: 'Low Timbale',
    67: 'High Agogo', 68: 'Low Agogo', 69: 'Cabasa', 70: 'Maracas',
    71: 'Short Whistle', 72: 'Long Whistle', 73: 'Short Guiro', 74: 'Long Guiro',
    75: 'Claves', 76: 'Hi Wood Block', 77: 'Low Wood Block', 78: 'Mute Cuica',
    79: 'Open Cuica', 80: 'Mute Triangle', 81: 'Open Triangle',
}

FAMILIES = ['Piano', 'Chromatic Perc.', 'Organ', 'Guitar', 'Bass', 'Strings', 'Ensemble',
            'Brass', 'Reed', 'Pipe', 'Synth Lead', 'Synth Pad', 'Synth FX', 'Ethnic',
            'Percussive', 'Sound FX']

# Feature-dict schema (extraction pipeline). Unknown fields are picked up too.
SCALAR_FIELDS = ('all_events', 'score_notes', 'score_chords', 'tracks',
                 'lyric_events', 'text_events', 'other_events', 'patch_change_events')
SUB_SCALARS = {'aligned': ('dtime_ms', 'aligned', 'total'),
               'karaoke': ('text', 'lyric'),
               'run_time': ('total', 'last_time'),
               'clean_midi': ('clean', 'total'),
               'dupe_pitches': ('deduped', 'total'),
               'bad_durs': ('bad', 'zero', 'total')}
BOOL_FIELDS = ('all_chords_good', 'text_lyric_latin')
COUNTER_FIELDS = ('features_counts', 'mono_mels', 'pitches_patches_counts')
_KNOWN_HANDLED = set(SCALAR_FIELDS) | set(BOOL_FIELDS) | set(COUNTER_FIELDS) | \
    set(SUB_SCALARS) | {'midi_path', 'bad_durs.counts', 'bad_durs.count'}

VALUE_ORDER = ['score_notes', 'all_events', 'run_time.total', 'run_time.last_time',
               'notes_per_sec', 'events_per_note', 'tracks', 'score_chords',
               'bars', 'bad_chords',
               'aligned.total', 'aligned.aligned', 'aligned.dtime_ms',
               'clean_midi.total', 'clean_midi.clean',
               'dupe_pitches.total', 'dupe_pitches.deduped',
               'bad_durs.total', 'bad_durs.bad', 'bad_durs.zero',
               'karaoke.text', 'karaoke.lyric', 'text_events', 'lyric_events',
               'other_events', 'patch_change_events',
               'ratio.clean', 'ratio.aligned', 'ratio.dupe_removed', 'ratio.bad_durs']

# ========================== streaming histogram machinery (defined BEFORE use)
LIN, LOG = 'lin', 'log'
_COUNT_SPEC = (LOG, 0.5, 1e8, 128)
_SPEC_TABLE = {
    'ratio.clean': (LIN, 0.0, 100.0, 100), 'ratio.aligned': (LIN, 0.0, 100.0, 100),
    'ratio.dupe_removed': (LIN, 0.0, 100.0, 100), 'ratio.bad_durs': (LIN, 0.0, 100.0, 100),
    'tracks': (LIN, 0, 24, 24),
    'events_per_note': (LOG, 0.25, 128, 96),
    'notes_per_sec': (LOG, 0.01, 1e4, 96),
    'aligned.dtime_ms': (LOG, 0.5, 1e4, 96),
    'run_time.total': (LOG, 100.0, 1e9, 128), 'run_time.last_time': (LOG, 100.0, 1e9, 128),
    'karaoke.text': (LOG, 0.5, 1e5, 64), 'karaoke.lyric': (LOG, 0.5, 1e5, 64),
    'text_events': (LOG, 0.5, 1e5, 64), 'lyric_events': (LOG, 0.5, 1e5, 64),
    'patch_change_events': (LOG, 0.5, 1e5, 64), 'other_events': (LOG, 0.5, 1e6, 96),
    'bars': (LOG, 0.5, 1e5, 96), 'bad_chords': (LOG, 0.5, 1e6, 96),
}
_FALLBACK_SPEC = (LOG, 0.1, 1e9, 128)
_DISCRETE_KEYS = {'aligned.dtime_ms', 'karaoke.text', 'karaoke.lyric',
                  'text_events', 'lyric_events', 'patch_change_events'}
_SCATTER_BINS = 50


def _spec_edges(spec):
    kind, lo, hi, nb = spec
    if kind == LOG:
        if HAVE_NUMPY:
            return [0.0] + np.logspace(math.log10(lo), math.log10(hi), nb).tolist()
        r = (hi / lo) ** (1.0 / (nb - 1))
        return [0.0] + [lo * (r ** i) for i in range(nb)]
    if HAVE_NUMPY:
        return np.linspace(lo, hi, nb).tolist()
    step = (hi - lo) / (nb - 1)
    return [lo + step * i for i in range(nb)]


_EDGE_CACHE = {}


def _spec_edges_cached(spec):
    key = (spec[0], spec[1], spec[2], spec[3])
    if key not in _EDGE_CACHE:
        _EDGE_CACHE[key] = _spec_edges(spec)
    return _EDGE_CACHE[key]


def _edges_for_key(key):
    spec = _SPEC_TABLE.get(key)
    if spec is None:
        spec = _COUNT_SPEC if any(key.startswith(p) for p in
                                  ('score_notes', 'all_events', 'aligned.', 'clean_midi.',
                                   'dupe_pitches.', 'bad_durs.')) else _FALLBACK_SPEC
    return _spec_edges_cached(spec)


class Hist:
    __slots__ = ('edges', 'counts', 'under', 'over', 'n')

    def __init__(self, edges):
        self.edges = edges
        self.counts = [0] * (len(edges) - 1)
        self.under = 0
        self.over = 0
        self.n = 0

    def add(self, v):
        self.n += 1
        if v < self.edges[0]:
            self.under += 1
        elif v > self.edges[-1]:
            self.over += 1
        else:
            i = bisect_right(self.edges, v) - 1
            if i < 0:
                self.under += 1
            elif i >= len(self.counts):
                self.over += 1
            else:
                self.counts[i] += 1

    def merge(self, other):
        self.n += other.n
        self.under += other.under
        self.over += other.over
        c = self.counts
        for i, x in enumerate(other.counts):
            c[i] += x


def hist_percentile(edges, counts, under, over, q):
    total = sum(counts) + under + over
    if total <= 0:
        return None
    t = total * q / 100.0
    if t <= under:
        return edges[0]
    cum = under
    for i, c in enumerate(counts):
        if cum + c >= t:
            lo, hi = edges[i], edges[i + 1]
            frac = (t - cum) / c if c else 0.0
            return lo + (hi - lo) * frac
        cum += c
    return edges[-1]


# Module-level pre-binned scatter axes (safe: machinery is defined above).
_SC_X = _spec_edges_cached((LOG, 1e3, 1e8, _SCATTER_BINS))   # play time, ms
_SC_Y = _spec_edges_cached((LOG, 1.0, 1e7, _SCATTER_BINS))   # notes


@dataclass
class Config:
    branding: str = DEFAULT_BRANDING
    attribution: str = DEFAULT_ATTRIBUTION
    output_dir: str = 'gmidi_stats'
    verbosity: int = 1            # 0 quiet · 1 normal · 2 verbose (full report) · 3 debug
    progress: bool = True
    workers: int = -1             # -1 = all cores · 1 = sequential
    backend: str = 'loky'         # loky | threading | multiprocessing | sequential
    batch_size: int = 20000       # feature dicts per parallel batch (large-dataset tuning)
    top_n: int = 15
    save_text_report: bool = True
    save_json: bool = True
    save_plots: bool = True
    show_plots: bool = False
    plot_format: str = 'png'
    dpi: int = 150
    cmap: str = 'plasma'
    # --- outlier guards (corrupt source data protection) --------------------
    # Per-file run_time is clipped to max_runtime_ms before entering ANY
    # statistic; per-file note density is clipped to max_density_nps. Clipped
    # files are counted and reported (nothing is silently hidden).
    max_runtime_ms: int = 86_400_000      # 24 h — real MIDIs never exceed this
    max_density_nps: float = 1000.0       # notes/sec — far above any real score


# =========================================================== small utilities
def _log(msg, level=1, cfg=None):
    if level == 0 or (cfg is not None and cfg.verbosity >= level):
        print(msg, flush=True)


class _Bar:
    def __init__(self, it): self.it = it
    def __iter__(self): return iter(self.it)


def _progress(it, cfg, desc='', total=None, leave=False):
    if cfg.progress and cfg.verbosity >= 1 and HAVE_TQDM:
        return tqdm(it, desc=desc, total=total, dynamic_ncols=True, leave=leave)
    return _Bar(it)


def run_parallel(fn, items, cfg, desc='Processing'):
    items = list(items)
    n = len(items)
    if n == 0:
        return []
    workers = cfg.workers if (cfg.workers and cfg.workers > 0) else (os.cpu_count() or 1)
    if cfg.backend == 'sequential':
        workers = 1
    if workers <= 1 or n < 2:
        return [fn(x) for x in _progress(items, cfg, desc, n)]
    if HAVE_JOBLIB:
        _log(f'  → {desc}: {n} item(s) · {workers} workers · backend={cfg.backend}', 2, cfg)
        return Parallel(n_jobs=workers, backend=cfg.backend)(
            delayed(fn)(x) for x in _progress(items, cfg, desc, n))
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    EX = ThreadPoolExecutor if cfg.backend == 'threading' else ProcessPoolExecutor
    out = []
    with EX(max_workers=workers) as ex:
        futs = [ex.submit(fn, x) for x in items]
        for f in _progress(as_completed(futs), cfg, desc, n):
            out.append(f.result())
    return out


def human_int(n):
    try:
        return f'{int(n):,}'
    except Exception:
        return str(n)


def human_big(n):
    """Compact formatting that stays readable into the billions (banners)."""
    n = float(n or 0)
    a = abs(n)
    if a >= 1e9:
        return f'{n / 1e9:.2f}B'
    if a >= 1e6:
        return f'{n / 1e6:.2f}M'
    if a >= 1e4:
        return f'{n / 1e3:.1f}K'
    return human_int(n)


def human_time_ms(ms):
    if ms is None:
        return '—'
    ms = float(ms)
    if ms < 1000:
        return f'{ms:.0f} ms'
    total_s = ms / 1000.0
    if total_s < 60:
        return f'{total_s:,.1f} s'
    m, s = divmod(total_s, 60)
    if m < 60:
        return f'{int(m)}m {s:04.1f}s'
    h, m = divmod(int(m), 60)
    return f'{h}h {m:02d}m {int(s):02d}s'


def human_duration_long(ms):
    """Scales gracefully to millions of files: hours → days → years of music."""
    s = float(ms or 0) / 1000.0
    yrs = s / (86400.0 * 365.25)
    if yrs >= 1:
        return f'≈ {yrs:,.1f} years of music'
    days = s / 86400.0
    if days >= 2:
        return f'≈ {days:,.1f} days of music'
    return human_time_ms(ms)


def bar(frac, width=14):
    frac = 0.0 if frac != frac else max(0.0, min(1.0, float(frac)))
    f = int(round(frac * width))
    return '█' * f + '░' * (width - f)


def _trunc(s, n):
    s = str(s)
    return s if len(s) <= n else s[:max(1, n - 1)] + '…'


def note_name(p):
    p = int(p)
    return f'{NOTE_NAMES[p % 12]}{p // 12 - 1}' if 0 <= p <= 127 else str(p)


def drum_name(p):
    """Percussion name for a drum-channel note (GM kit map)."""
    p = int(p)
    return GM_DRUMS.get(p, f'perc note {p}')


def pitch_label(p, patch=None):
    """Note name for melodic notes, percussion name for drum-channel notes."""
    p = int(p)
    if patch == 128:
        return f'{drum_name(p)} ({p})'
    return f'{note_name(p)} ({p})'


def drum_label(p):
    return f'{drum_name(p)} ({p})'


def patch_name(p):
    p = int(p)
    if p == 128:
        return 'Drums'
    if 0 <= p < 128:
        return GM_PATCHES[p]
    return f'patch {p}'


def patch_label(p):
    return f'{patch_name(p)} [{p}]'


def family_name(p):
    p = int(p)
    if p == 128:
        return 'Drums'
    return FAMILIES[p // 8] if 0 <= p < 128 else 'Sound FX'


def chord_name(idx):
    if HAVE_TMIDIX and hasattr(TMIDIX, 'ALL_CHORDS_SORTED'):
        try:
            return '-'.join(NOTE_NAMES[c % 12] for c in TMIDIX.ALL_CHORDS_SORTED[int(idx)])
        except Exception:
            pass
    return f'chord#{idx}'


def token_category(tok):
    if tok == 978: return 'bad_chords'
    if tok == 979: return 'bars'
    if 1 <= tok <= 127: return 'delta_time'
    if 129 <= tok <= 255: return 'duration'
    if 257 <= tok <= 383: return 'pitch'
    if 385 <= tok <= 511: return 'velocity'
    if 512 <= tok <= 640: return 'patch'
    if 641 <= tok <= 656: return 'channel'
    if 657 <= tok <= 977: return 'chord'
    return 'unknown'


def _norm_key(k):
    if isinstance(k, int):
        return k
    if isinstance(k, str):
        s = k.strip()
        if s.startswith('(') and s.endswith(')'):
            try:
                return tuple(int(x) for x in s[1:-1].split(','))
            except Exception:
                return k
        try:
            return int(s)
        except Exception:
            return k
    return k


def _add_counter(dst, d):
    """Fast dict-based counter merge (int/float values only, tuple keys restored)."""
    for k, v in d.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        k = _norm_key(k)
        dst[k] = dst.get(k, 0) + int(v)


def _mc(counter, n=None):
    """Top-N items from a Counter OR a plain dict (deterministic tie-break)."""
    if counter is None:
        return []
    if hasattr(counter, 'most_common'):
        items = counter.most_common()
    else:
        items = list(counter.items())
    items.sort(key=lambda kv: (-kv[1], str(kv[0])))
    return items[:n] if n is not None else items


def melodic_pitch_profile(stats):
    """(pitch -> count) for melodic (non-drum) notes; falls back to vocab tokens."""
    pp = (stats.get('counters') or {}).get('pitches_patches_counts') or {}
    mel = Counter()
    for k, v in pp.items():
        if isinstance(k, tuple) and len(k) == 2 and k[1] != 128:
            mel[k[0]] += v
    if not mel:
        for tok, v in ((stats.get('counters') or {}).get('features_counts') or {}).items():
            if isinstance(tok, int) and 257 <= tok <= 383:
                mel[tok - 256] += v
    return mel


def percussion_profile(stats):
    """(drum note -> hit count) from pitches_patches_counts."""
    pp = (stats.get('counters') or {}).get('pitches_patches_counts') or {}
    perc = Counter()
    for k, v in pp.items():
        if isinstance(k, tuple) and len(k) == 2 and k[1] == 128:
            perc[k[0]] += v
    return perc


# ============================================================ input loading
KNOWN_KEYS = ('score_notes', 'score_chords', 'features_counts', 'run_time', 'all_events',
              'pitches_patches_counts', 'mono_mels', 'clean_midi', 'dupe_pitches',
              'bad_durs', 'aligned', 'karaoke', 'tracks', 'text_lyric_latin',
              'all_chords_good', 'other_events', 'patch_change_events')


def _looks_like_feature_dict(d):
    if not isinstance(d, dict) or not d:
        return False
    return any(k in d for k in KNOWN_KEYS)


def _unwrap_item(item):
    """Normalize one raw input item → list of (uid, feature_dict) pairs.

    Accepts:
      • a bare feature dict                        → [(uid from midi_path, dict)]
      • a wrapper {md5_hash: feature_dict}         → [(md5_hash, dict)]
      • a mapping of several {uid: feature_dict}   → [(uid, dict), ...]
    Non-matching items yield [].
    """
    if not isinstance(item, dict):
        return []
    if _looks_like_feature_dict(item):
        uid = item.get('midi_path')
        return [(uid if isinstance(uid, str) and uid else None, item)]
    out = []
    for k, v in item.items():
        if isinstance(v, dict) and _looks_like_feature_dict(v):
            out.append((str(k), v))
    return out


def _load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return path, json.load(f)


def _md5_file(path, chunk=1 << 20):
    try:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(chunk), b''):
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def extract_features_from_midi(input_midi):
    """Single-MIDI feature extraction (TMIDIX pipeline) — parallel-friendly."""
    try:
        import TMIDIX
        from collections import Counter

        raw_score = TMIDIX.midi2single_track_ms_score(input_midi, do_not_check_MIDI_signature=True)
        data = TMIDIX.advanced_score_processor(raw_score,
                                               return_score_analysis=True,
                                               return_enhanced_score_notes=True,
                                               return_text_and_lyric_events=True,
                                               apply_sustain=True)
        if len(data) == 3:
            analysis, raw_escore_notes, text_events = data
        else:
            analysis, raw_escore_notes = data
            text_events = []
        if not raw_escore_notes:
            return None

        dscore = TMIDIX.delta_score_notes(raw_escore_notes, timings_clip_value=3999)
        dtimes = [e[1] for e in dscore if e[1] != 0]
        mc = Counter(dtimes).most_common(1)[0]
        aligned = {'dtime_ms': mc[0], 'aligned': mc[1], 'total': len(dtimes)}
        karaoke = {'text': len([e for e in text_events if e[0] == 'text_event']),
                   'lyric': len([e for e in text_events if e[0] == 'lyric'])}
        rt = TMIDIX.escore_notes_run_time(raw_escore_notes)
        run_time = {'total': rt[0], 'last_time': rt[1]}

        escore_notes = TMIDIX.augment_enhanced_score_notes(raw_escore_notes, timings_divider=32)
        clean = [e for e in escore_notes if e[6] in TMIDIX.CLEAN_INSTRUMENTS and e[3] != 9]
        clean_midi = {'clean': len(clean), 'total': len(escore_notes)}
        dd = TMIDIX.remove_duplicate_pitches_from_escore_notes(escore_notes)
        dupe_pitches = {'deduped': len(dd), 'total': len(escore_notes)}
        bds = TMIDIX.escore_notes_durations_counter(dd, min_duration=128)
        bad_durs = {'bad': bds[0], 'counts': bds[3] or {}, 'zero': bds[2], 'total': bds[1]}
        fixed = TMIDIX.fix_escore_notes_durations(dd, min_notes_gap=0)

        cscore = TMIDIX.chordify_score([1000, fixed])
        fixed_score, bad_chords_counter = [], 0
        for c in cscore:
            tones_chord = sorted(set([e[4] % 12 for e in c if e[3] != 9]))
            if tones_chord:
                if tones_chord not in TMIDIX.ALL_CHORDS_SORTED:
                    tones_chord = TMIDIX.check_and_fix_tones_chord(tones_chord, use_full_chords=False)
                    bad_chords_counter += 1
            for e in c:
                if e[4] % 12 in tones_chord or e[3] == 9:
                    fixed_score.append(e)

        mono_mels = dict(TMIDIX.escore_notes_monoponic_melodies(
            [e for e in fixed_score if e[3] != 9]))

        cscore = TMIDIX.chordify_score([1000, fixed_score])
        score, pp_counter = [], Counter()
        abs_time, pbar, bars_count, pc = 0, -1, 0, cscore[0]
        for c in cscore:
            if abs_time // 128 > pbar:
                bars_count += 1
                pbar = abs_time // 128
            tones_chord = sorted(set([e[4] % 12 for e in c if e[3] != 9]))
            if tones_chord and len(c) > 1:
                score.append(TMIDIX.ALL_CHORDS_SORTED.index(tones_chord) + 657)
            dtime = max(0, min(127, c[0][1] - pc[0][1]))
            if dtime != 0:
                score.append(dtime)
            abs_time += dtime
            for e in c:
                score.extend([max(1, min(127, e[2])) + 128, max(1, min(127, e[4])) + 256,
                              max(1, min(127, e[5])) + 384, max(0, min(128, e[6])) + 512,
                              max(0, min(15, e[3])) + 641])
                pp_counter[(max(1, min(127, e[4])), max(0, min(128, e[6])))] += 1
            pc = c

        features_counter = Counter(score)
        features_counter[978] = bad_chords_counter
        features_counter[979] = bars_count

        final_dict = {'midi_path': str(input_midi)}
        final_dict |= {k.lower().replace(' ', '_').replace('number_of_', ''): v for k, v in analysis}
        for k in ('ticks_per_quarter_note', 'shortest_chord', 'longest_chord',
                  'score_patches', 'score_pitches', 'score_tones', 'bad_chords'):
            final_dict.pop(k, None)
        final_dict['text_lyric_latin'] = final_dict.pop('all_text_and_lyric_events_latin', None)
        final_dict['aligned'] = aligned
        final_dict['karaoke'] = karaoke
        final_dict['run_time'] = run_time
        final_dict['clean_midi'] = clean_midi
        final_dict['dupe_pitches'] = dupe_pitches
        final_dict['bad_durs'] = bad_durs
        final_dict['mono_mels'] = mono_mels
        final_dict['features_counts'] = dict(features_counter.most_common())
        final_dict['pitches_patches_counts'] = dict(pp_counter.most_common())
        return (_md5_file(input_midi) or Path(input_midi).stem, final_dict)
    except Exception:
        if os.environ.get('GMIDI_DEBUG'):
            traceback.print_exc()
        return None


def collect_feature_dicts(inputs, cfg):
    json_files, midi_files = [], []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for q in sorted(p.rglob('*')):
                if q.is_file():
                    sfx = q.suffix.lower()
                    if sfx == '.json':
                        json_files.append(q)
                    elif sfx in ('.mid', '.midi'):
                        midi_files.append(q)
        elif p.suffix.lower() == '.json':
            json_files.append(p)
        elif p.suffix.lower() in ('.mid', '.midi'):
            midi_files.append(p)
        else:
            _log(f'! Skipping unrecognized input: {inp}', 0, cfg)
    json_files.sort()
    midi_files.sort()

    dicts, n_json, n_midi = {}, 0, 0
    if json_files:
        _log(f'Loading {len(json_files)} JSON file(s)…', 1, cfg)
        for path, data in run_parallel(_load_json_file, [str(p) for p in json_files],
                                       cfg, desc='Loading JSON'):
            if isinstance(data, dict) and not _looks_like_feature_dict(data) and data:
                it = [(str(k), v) for k, v in data.items()
                      if isinstance(v, dict) and _looks_like_feature_dict(v)]
            elif isinstance(data, dict):
                it = [(Path(path).stem, data)]
            elif isinstance(data, list):
                it = []
                for i, d in enumerate(data):
                    for uid, fd in _unwrap_item(d):
                        it.append((uid or f'{Path(path).stem}#{i}', fd))
            else:
                _log(f'! Unrecognized JSON structure: {path}', 0, cfg)
                continue
            for uid, fd in it:
                if isinstance(fd, dict):
                    dicts[str(uid)] = fd
                    n_json += 1

    if midi_files:
        if not HAVE_TMIDIX:
            _log('! TMIDIX not available — cannot extract features from MIDI files.', 0, cfg)
        else:
            _log(f'Extracting features from {len(midi_files)} MIDI file(s)…', 1, cfg)
            for r in run_parallel(extract_features_from_midi, [str(p) for p in midi_files],
                                  cfg, desc='Extracting MIDIs'):
                if r:
                    dicts[r[0]] = r[1]
                    n_midi += 1

    _log(f'✓ Collected {len(dicts):,} feature dicts '
         f'({n_json:,} from JSON, {n_midi:,} from MIDI).', 1, cfg)
    return dicts


# ===================================================== parallel aggregation
def _new_pack():
    return {'n': 0, 'sum': {}, 'sumsq': {}, 'min': {}, 'max': {}, 'hist': {},
            'disc': defaultdict(Counter), 'bools': {}, 'counters': {},
            'presence': Counter(), 'top_notes': [], 'top_time': [], 'scatter': None,
            'capped_runtime': 0, 'capped_density': 0}


def _add_val(pk, key, v):
    v = float(v)
    pk['sum'][key] = pk['sum'].get(key, 0.0) + v
    pk['sumsq'][key] = pk['sumsq'].get(key, 0.0) + v * v
    if key not in pk['min'] or v < pk['min'][key]:
        pk['min'][key] = v
    if key not in pk['max'] or v > pk['max'][key]:
        pk['max'][key] = v
    h = pk['hist'].get(key)
    if h is None:
        h = pk['hist'][key] = Hist(_edges_for_key(key))
    h.add(v)


def _agg_one(pk, fd, uid, cap=None, cap_nps=None):
    if not isinstance(fd, dict):
        return
    pk['n'] += 1

    for f in SCALAR_FIELDS:
        x = fd.get(f)
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            _add_val(pk, f, x)

    for parent, subs in SUB_SCALARS.items():
        sub = fd.get(parent)
        if isinstance(sub, dict):
            pk['presence'][parent] += 1
            capped_flag = False
            for s in subs:
                x = sub.get(s)
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    # --- outlier guard: clip insane runtimes (corrupt MIDIs) ---
                    if parent == 'run_time' and cap and x > cap:
                        x = float(cap)
                        capped_flag = True
                    _add_val(pk, parent + '.' + s, x)
            if capped_flag:
                pk['capped_runtime'] += 1
            if parent == 'bad_durs':
                bc = sub.get('counts', sub.get('count'))
                if isinstance(bc, dict) and bc:
                    c = pk['counters'].setdefault('bad_durs.counts', {})
                    _add_counter(c, bc)
                    pk['presence']['bad_durs.counts'] += 1

    for b in BOOL_FIELDS:
        pk['bools'].setdefault(b, Counter())[str(fd.get(b, '<missing>'))] += 1

    for cf in COUNTER_FIELDS:
        c = fd.get(cf)
        if isinstance(c, dict) and c:
            dst = pk['counters'].setdefault(cf, {})
            _add_counter(dst, c)
            pk['presence'][cf] += 1

    for key in _DISCRETE_KEYS:                     # exact small-value counters
        if '.' in key:
            parent, sub_k = key.split('.', 1)
            sub = fd.get(parent)
            v = sub.get(sub_k) if isinstance(sub, dict) else None
        else:
            v = fd.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            dc = pk['disc'][key]
            dc[int(v)] += 1
            if len(dc) > 4096:                     # bound memory on pathological data
                pk['disc'][key] = Counter()

    # ---- derived per-file metrics --------------------------------------
    cm = fd.get('clean_midi') if isinstance(fd.get('clean_midi'), dict) else {}
    dp = fd.get('dupe_pitches') if isinstance(fd.get('dupe_pitches'), dict) else {}
    bd = fd.get('bad_durs') if isinstance(fd.get('bad_durs'), dict) else {}
    al = fd.get('aligned') if isinstance(fd.get('aligned'), dict) else {}
    rt = fd.get('run_time') if isinstance(fd.get('run_time'), dict) else {}
    fc = fd.get('features_counts') if isinstance(fd.get('features_counts'), dict) else {}
    sn, ae = fd.get('score_notes'), fd.get('all_events')

    cm_t = cm.get('total')
    if isinstance(cm_t, (int, float)) and cm_t:
        _add_val(pk, 'ratio.clean', 100.0 * (cm.get('clean') or 0) / cm_t)
    dp_t = dp.get('total')
    if isinstance(dp_t, (int, float)) and dp_t:
        _add_val(pk, 'ratio.dupe_removed', 100.0 * (dp_t - (dp.get('deduped') or 0)) / dp_t)
    bd_t = bd.get('total')
    if isinstance(bd_t, (int, float)) and bd_t:
        _add_val(pk, 'ratio.bad_durs',
                 100.0 * ((bd.get('bad') or 0) + (bd.get('zero') or 0)) / bd_t)
    al_t = al.get('total')
    if isinstance(al_t, (int, float)) and al_t:
        _add_val(pk, 'ratio.aligned', 100.0 * (al.get('aligned') or 0) / al_t)

    # runtime (clipped) drives density, scatter and top-file lists
    rt_t = rt.get('total')
    if isinstance(rt_t, (int, float)) and rt_t > 0:
        rt_t = float(rt_t)
        if cap and rt_t > cap:
            rt_t = float(cap)
        if isinstance(sn, (int, float)):
            nps = 1000.0 * sn / rt_t
            if cap_nps and nps > cap_nps:          # tiny/corrupt runtime guard
                nps = float(cap_nps)
                pk['capped_density'] += 1
            _add_val(pk, 'notes_per_sec', nps)
    if isinstance(sn, (int, float)) and sn > 0 and isinstance(ae, (int, float)):
        _add_val(pk, 'events_per_note', ae / sn)
    if fc:
        bars = fc.get(979, 0)
        badc = fc.get(978, 0)
        if bars:
            _add_val(pk, 'bars', bars)
        if badc:
            _add_val(pk, 'bad_chords', badc)

    # ---- forward-compat: unknown fields ---------------------------------
    for k, v in fd.items():
        if k in _KNOWN_HANDLED:
            continue
        if isinstance(v, bool):
            pk['bools'].setdefault(k, Counter())[str(v)] += 1
        elif isinstance(v, (int, float)):
            _add_val(pk, k, v)
        elif isinstance(v, dict) and v:
            if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v.values()):
                for sk, sv in v.items():
                    if isinstance(sv, (int, float)) and not isinstance(sv, bool):
                        _add_val(pk, f'{k}.{sk}', sv)
            else:
                dst = pk['counters'].setdefault(k, {})
                _add_counter(dst, v)
                pk['presence'][k] += 1

    # ---- notable files (bounded top-K) ----------------------------------
    sn_i = int(sn) if isinstance(sn, (int, float)) else 0
    ms_f = float(rt_t) if isinstance(rt_t, (int, float)) and rt_t > 0 else 0.0
    ae_i = int(ae) if isinstance(ae, (int, float)) else 0
    entry = (sn_i, uid, ms_f, ae_i)
    pk['top_notes'].append(entry)
    pk['top_time'].append(entry)

    # ---- density map (notes vs play time) --------------------------------
    if sn_i > 0 and ms_f > 0:
        sc = pk['scatter']
        if sc is None:
            sc = pk['scatter'] = [[0] * _SCATTER_BINS for _ in range(_SCATTER_BINS)]
        xi = bisect_right(_SC_X, ms_f) - 1
        yi = bisect_right(_SC_Y, sn_i) - 1
        if 0 <= xi < _SCATTER_BINS and 0 <= yi < _SCATTER_BINS:
            sc[yi][xi] += 1


def _aggregate_chunk(chunk, start=0, cap=None, cap_nps=None):
    """Worker: aggregate a list of feature dicts (bare OR {hash: dict}-wrapped)
    → mergeable partial pack."""
    pk = _new_pack()
    idx = start
    for item in chunk:
        pairs = _unwrap_item(item)
        if not pairs:
            idx += 1
            continue
        for uid, fd in pairs:
            if uid is None:
                mp = fd.get('midi_path')
                uid = mp if isinstance(mp, str) and mp else f'#{idx}'
            _agg_one(pk, fd, str(uid), cap, cap_nps)
            idx += 1
    pk['top_notes'] = sorted(pk['top_notes'], key=lambda t: -t[0])[:50]
    pk['top_time'] = sorted(pk['top_time'], key=lambda t: -t[2])[:50]
    return pk


def _merge_pack(a, b):
    a['n'] += b['n']
    for k, v in b['sum'].items():
        a['sum'][k] = a['sum'].get(k, 0.0) + v
    for k, v in b['sumsq'].items():
        a['sumsq'][k] = a['sumsq'].get(k, 0.0) + v
    for k, v in b['min'].items():
        if k not in a['min'] or v < a['min'][k]:
            a['min'][k] = v
    for k, v in b['max'].items():
        if k not in a['max'] or v > a['max'][k]:
            a['max'][k] = v
    for k, h in b['hist'].items():
        if k in a['hist']:
            a['hist'][k].merge(h)
        else:
            a['hist'][k] = h
    for k, c in b['disc'].items():
        a['disc'][k].update(c)
    for k, c in b['bools'].items():
        a['bools'].setdefault(k, Counter()).update(c)
    for k, c in b['counters'].items():
        dst = a['counters'].setdefault(k, {})
        for kk, vv in c.items():
            dst[kk] = dst.get(kk, 0) + vv
    a['presence'].update(b['presence'])
    a['capped_runtime'] += b.get('capped_runtime', 0)
    a['capped_density'] += b.get('capped_density', 0)
    a['top_notes'] = sorted(a['top_notes'] + b['top_notes'], key=lambda t: -t[0])[:50]
    a['top_time'] = sorted(a['top_time'] + b['top_time'], key=lambda t: -t[2])[:50]
    if b['scatter'] is not None:
        if a['scatter'] is None:
            a['scatter'] = [[0] * _SCATTER_BINS for _ in range(_SCATTER_BINS)]
        for yi in range(_SCATTER_BINS):
            ra, rb = a['scatter'][yi], b['scatter'][yi]
            for xi in range(_SCATTER_BINS):
                ra[xi] += rb[xi]
    return a


def finalize_stats(pk, cfg):
    stats = {'n_files': pk['n'], 'values': {}, 'bools': pk['bools'],
             'counters': pk['counters'], 'discrete': dict(pk['disc']),
             'presence': pk['presence'], 'top_files': {'notes': pk['top_notes'],
                                                       'time': pk['top_time']},
             'scatter2d': {'x_edges': _SC_X, 'y_edges': _SC_Y, 'counts': pk['scatter']},
             'capped_runtime': pk.get('capped_runtime', 0),
             'capped_density': pk.get('capped_density', 0),
             'runtime_cap': cfg.max_runtime_ms,
             'density_cap': cfg.max_density_nps,
             'generated': datetime.now().isoformat(timespec='seconds'),
             'branding': cfg.branding, 'attribution': cfg.attribution}
    for key, h in pk['hist'].items():
        n = h.n
        s = pk['sum'].get(key, 0.0)
        mean = s / n if n else float('nan')
        var = max(0.0, (pk['sumsq'].get(key, 0.0) / n - mean * mean)) if n else 0.0
        st = {'n': n, 'sum': s, 'min': pk['min'].get(key), 'max': pk['max'].get(key),
              'mean': mean, 'std': math.sqrt(var),
              'hist': {'edges': h.edges, 'counts': h.counts, 'under': h.under, 'over': h.over}}
        for p in (5, 25, 50, 75, 95):
            st[f'p{p}'] = hist_percentile(h.edges, h.counts, h.under, h.over, p)
        stats['values'][key] = st
    return stats


def aggregate_stats(obj, cfg):
    """Aggregate ANY of: list of dicts · dict-of-dicts · indexable sequence · iterator.
    Items may be bare feature dicts or {md5_hash: feature_dict} wrappers.
    Progress: tqdm tracks *completed* batches (results as they arrive), so the
    bar moves in step with actual work under joblib/executor parallelism."""
    workers = cfg.workers if (cfg.workers and cfg.workers > 0) else (os.cpu_count() or 1)
    if cfg.backend == 'sequential':
        workers = 1
    cap = cfg.max_runtime_ms if cfg.max_runtime_ms else None
    cap_nps = cfg.max_density_nps if cfg.max_density_nps else None

    if isinstance(obj, dict):
        seq = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        seq = obj
    elif hasattr(obj, '__len__') and hasattr(obj, '__getitem__'):
        seq = obj
    else:
        seq = None                                   # one-shot iterator → streaming

    pack = _new_pack()

    if seq is None:
        _log('Input is a one-shot iterator — streaming sequentially…', 1, cfg)
        bs = max(1000, cfg.batch_size)
        batch, start, done = [], 0, 0
        for fd in _progress(obj, cfg, 'Aggregating', leave=True):
            batch.append(fd)
            if len(batch) >= bs:
                _merge_pack(pack, _aggregate_chunk(batch, start, cap, cap_nps))
                done += len(batch)
                start += len(batch)
                batch = []
        if batch:
            _merge_pack(pack, _aggregate_chunk(batch, start, cap, cap_nps))
            done += len(batch)
        _log(f'✓ Aggregated {done:,} feature dicts (streaming).', 1, cfg)
        return finalize_stats(pack, cfg)

    n = len(seq)
    if n == 0:
        raise ValueError('No feature dicts provided (empty input).')

    # Auto-tune batch size for a smooth progress bar (≥ ~100 ticks), without
    # shrinking batches on huge datasets (your batch_size cap still wins).
    target_ticks = max(4 * workers, 100)
    bs = min(max(1000, cfg.batch_size),
             max(1000, math.ceil(n / target_ticks)))
    batches = [(s, min(s + bs, n)) for s in range(0, n, bs)]
    nb = len(batches)

    if workers <= 1 or nb < 2:
        for s, e in _progress(batches, cfg, 'Aggregating', nb, leave=True):
            _merge_pack(pack, _aggregate_chunk(seq[s:e], s, cap, cap_nps))
    else:
        _log(f'Aggregating {n:,} feature dicts · {nb} batches · '
             f'{workers} workers · backend={cfg.backend}', 1, cfg)
        ok = False
        if HAVE_JOBLIB:
            # Stream results as they finish → tqdm tracks real completion.
            # ('generator_unordered' needs joblib ≥ 1.3, 'generator' ≥ 1.2.)
            for ret_as in ('generator_unordered', 'generator'):
                try:
                    results = Parallel(n_jobs=workers, backend=cfg.backend,
                                       return_as=ret_as)(
                        delayed(_aggregate_chunk)(seq[s:e], s, cap, cap_nps)
                        for s, e in batches)
                    for pk in _progress(results, cfg, 'Aggregating', nb, leave=True):
                        _merge_pack(pack, pk)
                    ok = True
                    break
                except (TypeError, ValueError):
                    continue                          # joblib too old → next mode
                except Exception as e:
                    _log(f'! Parallel aggregation failed ({e}) — '
                         f'falling back to executor mode.', 0, cfg)
                    pack = _new_pack()                # discard partial merges
                    break
        if not ok:
            from concurrent.futures import (ProcessPoolExecutor,
                                            ThreadPoolExecutor, as_completed)
            EX = ThreadPoolExecutor if cfg.backend == 'threading' else ProcessPoolExecutor
            try:
                with EX(max_workers=workers) as ex:
                    futs = [ex.submit(_aggregate_chunk, seq[s:e], s, cap, cap_nps)
                            for s, e in batches]
                    for f in _progress(as_completed(futs), cfg, 'Aggregating', nb,
                                       leave=True):
                        _merge_pack(pack, f.result())
            except Exception as e:
                _log(f'! Parallel aggregation failed ({e}) — sequential fallback.', 0, cfg)
                pack = _new_pack()
                for s, e in _progress(batches, cfg, 'Aggregating', nb, leave=True):
                    _merge_pack(pack, _aggregate_chunk(seq[s:e], s, cap, cap_nps))

    stats = finalize_stats(pack, cfg)
    _log(f'✓ Aggregated {stats["n_files"]:,} feature dicts.', 1, cfg)
    return stats


# ================================================================= reports
def sv(key, v):
    if v is None:
        return '—'
    if key.startswith('ratio.'):
        return f'{v:,.1f}%'
    if key.startswith('run_time.') or key == 'aligned.dtime_ms':
        return human_time_ms(v)
    if key in ('notes_per_sec', 'events_per_note'):
        return f'{v:,.2f}'
    if abs(v - round(v)) < 1e-9:
        return human_int(v)
    return f'{v:,.2f}'


def _bool_line(stats, field):
    c = stats['bools'].get(field) or {}
    tot = sum(c.values())
    if not tot:
        return '—'
    return ' · '.join(f'{k}: {v:,} ({100 * v / tot:.1f}%)' for k, v in c.most_common())


def build_text_report(stats, cfg):
    W = REPORT_WIDTH
    L = []
    V, DISC = stats['values'], stats.get('discrete', {})

    def fit(s, w):
        s = str(s)
        return s if len(s) <= w else s[:max(1, w - 1)] + '…'

    def kv(label, value):
        label, value = '  ' + str(label), str(value)
        label = fit(label, max(4, W - len(value) - 6))
        dots = max(2, W - len(label) - len(value) - 2)
        L.append(fit(f'{label} {"." * dots} {value}', W))

    sec_no = [0]

    def section(title):
        sec_no[0] += 1
        L.append('')
        L.append('━' * W)
        L.append(f'  {sec_no[0]} · {title.upper()}')
        L.append('━' * W)

    def top_block(title, counter, labeler=None, n=None, total=None):
        n = n or cfg.top_n
        total = total if total is not None else (sum(counter.values()) if counter else 0)
        L.append('')
        L.append(f'  {title}  —  total {human_int(total)}')
        if not counter:
            L.append('    (none)')
            return
        for i, (k, v) in enumerate(_mc(counter, n)):
            lab = fit(labeler(k) if labeler else str(k), 42)
            frac = (v / total) if total else 0.0
            L.append(f'    {i + 1:>3}. {lab:<42} {human_int(v):>13}  {bar(frac, 14)} '
                     f'{100 * frac:5.1f}%')

    S = lambda k: V.get(k, {})
    SUM = lambda k: S(k).get('sum', 0.0)

    # ---------------- header (strict width, no overflow) -----------------
    L.append('╔' + '═' * (W - 2) + '╗')
    L.append('║' + fit(f' {cfg.branding.upper()} — FEATURE STATISTICS REPORT', W - 2)
             .ljust(W - 2) + '║')
    L.append('║' + fit(f' {cfg.attribution}', W - 2).ljust(W - 2) + '║')
    L.append('╚' + '═' * (W - 2) + '╝')
    kv('Report generated', stats['generated'])
    kv('Feature dicts analyzed', human_int(stats['n_files']))

    # ---------------- 1 · overview ---------------------------------------
    section('Dataset overview')
    kv('Total notes', human_int(SUM('score_notes')))
    kv('Total MIDI events', human_int(SUM('all_events')))
    kv('Total play time', f'{human_time_ms(SUM("run_time.total"))}  '
                          f'({human_duration_long(SUM("run_time.total"))})')
    kv('Mean play time / file', sv('run_time.total', S('run_time.total').get('mean')))
    kv('Mean notes / file', f'{S("score_notes").get("mean", 0):,.1f}')
    kv('Median notes / file', sv('score_notes', S('score_notes').get('p50')))
    kv('Mean density', f'{S("notes_per_sec").get("mean", 0):,.2f} notes/sec')
    kv('Total bars', human_int(SUM('bars')))
    mm = stats['counters'].get('mono_mels') or {}
    if mm:
        kv('Total monophonic melody notes', human_int(sum(mm.values())))
    kv('Files, all chords good', _bool_line(stats, 'all_chords_good'))
    kv('Files, latin text/lyrics', _bool_line(stats, 'text_lyric_latin'))

    # ---------------- 2 · quality -----------------------------------------
    section('Quality & integrity')
    cm_t, cm_c = SUM('clean_midi.total'), SUM('clean_midi.clean')
    drums = sum(v for k, v in (stats['counters'].get('pitches_patches_counts')
                               or {}).items()
                if isinstance(k, tuple) and len(k) == 2 and k[1] == 128)
    if cm_t:
        other = max(0.0, cm_t - cm_c - drums)
        w = 34
        seg = [('█', cm_c), ('▓', drums), ('░', other)]
        comp = ''.join(ch * max(0, int(round(v / cm_t * w))) for ch, v in seg)
        kv('Note composition (█ melodic / ▓ drums / ░ other)', comp)
        L.append(f'    █ melodic (clean) {human_int(cm_c):>14}  {100 * cm_c / cm_t:5.1f}%   '
                 f'(melodic lead+bass patches)')
        L.append(f'    ▓ drums            {human_int(drums):>14}  {100 * drums / cm_t:5.1f}%   '
                 f'(drum channel, patch 128)')
        L.append(f'    ░ other            {human_int(other):>14}  {100 * other / cm_t:5.1f}%   '
                 f'(ethnic, percussive, SFX patches)')
    if stats.get('capped_runtime'):
        kv('Runtime outliers capped', f'{stats["capped_runtime"]:,} file(s) above '
                                      f'{human_time_ms(stats.get("runtime_cap") or 0)} '
                                      f'— clipped for aggregation')
    if stats.get('capped_density'):
        kv('Density outliers capped', f'{stats["capped_density"]:,} file(s) above '
                                      f'{stats.get("density_cap") or 0:,.0f} notes/sec '
                                      f'— clipped for aggregation')
    dp_t, dp_d = SUM('dupe_pitches.total'), SUM('dupe_pitches.deduped')
    if dp_t:
        dr = dp_t - dp_d
        kv('Duplicate-pitch notes removed', f'{human_int(dr)} ({100 * dr / dp_t:.2f}%)')
    bd_t = SUM('bad_durs.total')
    if bd_t:
        kv('Bad durations', f'{human_int(SUM("bad_durs.bad"))} '
                            f'({100 * SUM("bad_durs.bad") / bd_t:.2f}%) · '
                            f'zero: {human_int(SUM("bad_durs.zero"))}')
    kv('Chords fixed by chord-checker (total)', f'{human_int(SUM("bad_chords"))}  '
       f'(mean {S("bad_chords").get("mean", 0):,.1f} / file)')
    kv('Mean alignment share (dominant dtime)', sv('ratio.aligned', S('ratio.aligned').get('mean')))
    kv('Mean melodic (clean) share / file', sv('ratio.clean', S('ratio.clean').get('mean')))

    # ---------------- 3 · per-file distributions ---------------------------
    section('Per-file distributions')
    keys = [k for k in VALUE_ORDER if k in V] + sorted(k for k in V if k not in VALUE_ORDER)
    hdr = (f'  {"metric":<26}{"mean":>12}{"median":>12}{"p5":>10}'
           f'{"p95":>11}{"min":>11}{"max":>11}')
    L.append(hdr)
    L.append('  ' + '-' * (W - 2))
    for k in keys:
        st = S(k)
        if not st:
            continue
        row = (f'  {fit(k, 26):<26}{sv(k, st["mean"]):>12}{sv(k, st["p50"]):>12}'
               f'{sv(k, st["p5"]):>10}{sv(k, st["p95"]):>11}'
               f'{sv(k, st["min"]):>11}{sv(k, st["max"]):>11}')
        L.append(fit(row, W))

    # ---------------- 4 · timing & alignment -------------------------------
    section('Timing & alignment')
    dt = DISC.get('aligned.dtime_ms') or Counter()
    top_block('Dominant inter-chord delta-time per file', dt, lambda k: f'{k} ms', n=8)
    L.append('')
    L.append('  (delta-time & duration tokens below are in dataset units: '
             'original timing ÷ 32, i.e. 1 unit ≈ 32 ms)')
    FCC = stats['counters'].get('features_counts') or {}
    CATS = defaultdict(Counter)
    for tok, v in FCC.items():
        CATS[token_category(tok)][tok] += v
    top_block('Top delta-time tokens', CATS.get('delta_time', Counter()),
              lambda k: f'{k} (≈ {k * 32} ms)')

    # ---------------- 5 · pitches & instruments ----------------------------
    section('Pitches & instruments')
    pp = stats['counters'].get('pitches_patches_counts') or {}
    by_pitch, by_patch, by_perc = Counter(), Counter(), Counter()
    for k, v in pp.items():
        if isinstance(k, tuple) and len(k) == 2:
            by_patch[k[1]] += v
            if k[1] == 128:
                by_perc[k[0]] += v
            else:
                by_pitch[k[0]] += v
    top_block('Top pitches (melodic instruments)', by_pitch,
              lambda k: f'{note_name(k)} ({k})')
    top_block('Top percussion hits (drum channel, GM kit map)', by_perc, drum_label)
    top_block('Top instruments (General MIDI patches)', by_patch, patch_label)
    fam = Counter()
    for p, v in by_patch.items():
        fam[family_name(p)] += v
    top_block('Instrument families', fam)
    mm = stats['counters'].get('mono_mels') or {}
    top_block('Monophonic melody instruments (mono_mels)', mm, patch_label)
    top_block('Top pitch–instrument combinations', pp,
              lambda k: f'{pitch_label(k[0], k[1])} @ {patch_label(k[1])}'
              if isinstance(k, tuple) and len(k) == 2 else str(k), n=10)

    # ---------------- 6 · harmony ------------------------------------------
    section('Harmony')
    top_block('Top chord tokens (vocab 657+)', CATS.get('chord', Counter()),
              lambda k: f'#{k - 657} {chord_name(k - 657)}')
    bd = stats['counters'].get('bad_durs.counts') or {}
    if bd:
        top_block('Bad duration values (min_duration=128)', bd, lambda k: f'dur {k}', n=10)

    # ---------------- 7 · feature-token vocabulary --------------------------
    section('Feature-token vocabulary')
    if not FCC:
        L.append('  (no features_counts data)')
    else:
        total_tok = sum(FCC.values())
        kv('Total token occurrences', human_int(total_tok))
        L.append('  Structure: every note emits 5 tokens (duration, pitch, velocity, '
                 'patch, channel); each chord step adds a delta-time; multi-note '
                 'chords add one chord token.')
        L.append('')
        L.append(f'  {"category":<14}{"occurrences":>15}{"share":>9}{"distinct":>11}'
                 f'{"tokens":>20}')
        L.append('  ' + '-' * (W - 2))
        for cat in ('delta_time', 'duration', 'pitch', 'velocity', 'patch', 'channel', 'chord'):
            c = CATS.get(cat)
            if not c:
                continue
            t = sum(c.values())
            L.append(f'  {cat:<14}{human_int(t):>15}{100 * t / total_tok:8.1f}%'
                     f'{len(c):>11}{bar(t / total_tok, 14):>20}')
        for meta in ('bad_chords', 'bars'):
            c = CATS.get(meta)
            if c:
                kv(f'{meta} (meta token)', f'{human_int(sum(c.values()))} total')

    # ---------------- 8 · text & karaoke ------------------------------------
    section('Text & karaoke')
    kv('Text events (total)', human_int(SUM('karaoke.text')))
    kv('Lyric events (total)', human_int(SUM('karaoke.lyric')))
    kv('Other events (total)', human_int(SUM('other_events')))
    kv('Patch-change events (total)', human_int(SUM('patch_change_events')))

    # ---------------- 9 · notable files --------------------------------------
    section('Notable files')
    for title, key in (('By note count', 'notes'), ('By play time', 'time')):
        rows = stats.get('top_files', {}).get(key) or []
        if not rows:
            continue
        L.append(f'  {title}:')
        L.append(f'  {"#":>3}  {"file":<24}{"notes":>10}{"play time":>12}{"events":>10}{"notes/s":>9}')
        L.append('  ' + '-' * 70)
        for i, (n_, uid, ms, ev) in enumerate(rows[:10]):
            nps = f'{n_ / (ms / 1000.0):,.2f}' if ms else '—'
            L.append(f'  {i + 1:>3}  {fit(uid, 24):<24}{fit(human_int(n_), 10):>10}'
                     f'{fit(human_time_ms(ms), 12):>12}{fit(human_int(ev), 10):>10}{nps:>9}')
        L.append('')

    # ---------------- footer ---------------------------------------------------
    L.append('━' * W)
    kv('Outputs', f'{cfg.output_dir}/  (report.txt · stats_summary.json · plots/)')
    L.append('')
    L.append(fit(f'{cfg.branding} — {cfg.attribution}', W).center(W))
    return '\n'.join(L)


def save_json_summary(stats, path):
    out = {'branding': stats['branding'], 'attribution': stats['attribution'],
           'generated': stats['generated'], 'n_files': stats['n_files'],
           'capped_runtime': stats.get('capped_runtime', 0),
           'capped_density': stats.get('capped_density', 0),
           'runtime_cap_ms': stats.get('runtime_cap'),
           'density_cap_nps': stats.get('density_cap'),
           'values': {k: {kk: vv for kk, vv in st.items() if kk != 'hist'} | {
               'hist': {'edges': st['hist']['edges'][::2],
                        'counts': [sum(st['hist']['counts'][i:i + 2])
                                   for i in range(0, len(st['hist']['counts']), 2)],
                        'under': st['hist']['under'], 'over': st['hist']['over']}}
               for k, st in stats['values'].items()},
           'discrete': {k: dict(c.most_common(256)) for k, c in stats.get('discrete', {}).items()},
           'booleans': {k: dict(c) for k, c in stats['bools'].items()},
           'counters': {k: {str(kk): vv for kk, vv in
                            sorted(c.items(), key=lambda x: -x[1])[:500]}
                        for k, c in stats['counters'].items()},
           'presence': dict(stats['presence']),
           'top_files': {k: v[:100] for k, v in stats.get('top_files', {}).items()},
           'density_map': stats.get('scatter2d', {}).get('counts')}
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')


# ==================================================================== plots
_C1, _C2, _C3, _C4 = '#5f27cd', '#00b894', '#e17055', '#0984e3'
_DARK = '#0d1220'


def _cmap(name):
    try:
        return matplotlib.colormaps[name]
    except Exception:
        return plt.get_cmap(name)


def _setup_style():
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': '#fbfbfd',
        'axes.edgecolor': '#d2dae2', 'axes.grid': True,
        'grid.color': '#dfe6e9', 'grid.alpha': 0.6, 'grid.linestyle': '--', 'grid.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 10, 'axes.titlesize': 11.5, 'axes.labelsize': 9.5,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
    })


def _new_fig(cfg, nrows=1, ncols=1, figsize=None):
    if figsize is None:
        figsize = (7.2 * ncols, 5.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    return fig, [a for row in axes for a in row]


def _finish(fig, cfg, title, subtitle=None):
    fig.suptitle(title, fontsize=14, fontweight='bold', color='#1e272e')
    if subtitle:
        fig.text(0.5, 0.938, subtitle, ha='center', fontsize=9, color='#57606f')
    fig.text(0.01, 0.006, cfg.branding, fontsize=7, color='#9aa0a6')
    fig.text(0.99, 0.006, cfg.attribution, fontsize=7, color='#9aa0a6', ha='right')


def _save_fig(fig, cfg, outdir, name):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f'{name}.{cfg.plot_format}'
    fig.savefig(path, dpi=cfg.dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    if cfg.show_plots:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)
    _log(f'  ✓ saved {path}', 3, cfg)
    return str(path)


def _no_data(ax, title):
    ax.axis('off')
    ax.set_title(f'{title} (no data)', fontsize=11)


def _hist_panel(ax, st, title, xlabel, color, transform=None, logx=False):
    """Draw a histogram panel directly from streaming histogram stats."""
    if not st:
        _no_data(ax, title)
        return
    edges = np.asarray(st['hist']['edges'], dtype=float)
    counts = np.asarray(st['hist']['counts'], dtype=float)
    scale = 1.0 / 60000.0 if transform == 'minutes' else 1.0
    if transform == 'minutes':
        edges = edges * scale
    if counts.sum() <= 0:
        _no_data(ax, title)
        return
    if logx and edges[0] <= 0:                    # log axis cannot start at 0
        edges, counts = edges[1:], counts[1:]
        if counts.size == 0 or counts.sum() <= 0:
            _no_data(ax, title)
            return
    ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge',
           color=color, edgecolor='white', linewidth=0.3, alpha=0.92)
    if logx:
        ax.set_xscale('log')
    mean_v = st['mean'] * scale
    med_v = st['p50'] * scale
    ax.axvline(mean_v, color='#d63031', ls='--', lw=1.3, label=f'mean {mean_v:,.1f}')
    ax.axvline(med_v, color='#00b894', ls='-.', lw=1.3, label=f'median {med_v:,.1f}')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('files')
    ax.legend(frameon=False)


def plot_title_card(stats, cfg, outdir):
    """Square branded title card — bright starfield, glow kept clear of text."""
    import matplotlib.patheffects as pe
    rng = random.Random(2026)
    fig = plt.figure(figsize=(9, 9))
    fig.patch.set_facecolor(_DARK)
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_facecolor(_DARK)

    # --- nebula glow, centered high so the lower third stays dark for text
    N = 600
    xs = np.linspace(-1.05, 1.05, N)
    X, Y = np.meshgrid(xs, xs)
    R = np.sqrt(X ** 2 + (Y - 0.34) ** 2)
    glow = np.exp(-((R / 0.46) ** 2.4))
    ax.imshow(glow, extent=(-1.05, 1.05, -1.05, 1.05), origin='lower',
              cmap=cfg.cmap, alpha=0.95, aspect='equal')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis('off')

    # --- orbit rings around the glow
    for r in (0.42, 0.58, 0.74, 0.90):
        ax.add_patch(plt.Circle((0, 0.34), r, fill=False, color='white',
                                alpha=0.10, lw=0.8))

    # --- stars: two passes (dim field + bright accents), clearly visible;
    #     kept sparse around the caption zone in the lower third
    def star_field(n, smin, smax, amin, amax):
        made = 0
        while made < n:
            a = rng.uniform(0, 2 * math.pi)
            r = math.sqrt(rng.uniform(0, 1)) * 1.05
            x, y = r * math.cos(a), r * math.sin(a)
            if -0.66 < y < -0.16 and abs(x) < 0.62 and rng.random() < 0.85:
                continue
            ax.scatter(x, y, s=rng.uniform(smin, smax), color='white',
                       alpha=rng.uniform(amin, amax), zorder=3, lw=0)
            made += 1
    star_field(210, 1.0, 5.0, 0.25, 0.75)
    star_field(70, 6.0, 16.0, 0.7, 1.0)
    for _ in range(14):                             # a few colored accent stars
        a = rng.uniform(0, 2 * math.pi)
        r = math.sqrt(rng.uniform(0.05, 1.0)) * 1.0
        ax.scatter(r * math.cos(a), 0.34 + r * math.sin(a),
                   s=rng.uniform(10, 26), lw=0, zorder=3, alpha=0.85,
                   color=_cmap(cfg.cmap)(rng.uniform(0.55, 0.95)))

    # --- text: big number lives in the dark lower third, with a dark stroke
    #     halo so it stays readable no matter what is behind it
    stroke = [pe.withStroke(linewidth=5, foreground=_DARK)]
    notes = stats['values'].get('score_notes', {}).get('sum', 0)
    ms = stats['values'].get('run_time.total', {}).get('sum', 0)
    fig.text(0.5, 0.945, cfg.branding, ha='center', fontsize=30, fontweight='bold',
             color='white', path_effects=stroke)
    fig.text(0.5, 0.905, 'F E A T U R E   S T A T I S T I C S', ha='center',
             fontsize=10.5, color='#c8cdd6', path_effects=stroke)
    ax.text(0, -0.30, human_big(notes), ha='center', va='center', fontsize=42,
            fontweight='bold', color='white', zorder=6, path_effects=stroke)
    ax.text(0, -0.425, 'NOTES', ha='center', va='center', fontsize=12,
            color='#dfe3ea', zorder=6, path_effects=stroke)
    ax.text(0, -0.55, f'{human_big(stats["n_files"])} files   ·   {human_duration_long(ms)}',
            ha='center', va='center', fontsize=11, color='#e8e8ee', zorder=6,
            path_effects=stroke)
    fig.text(0.5, 0.058, cfg.attribution, ha='center', fontsize=9, color='#9aa0a6')
    fig.text(0.5, 0.034, f'generated {stats["generated"]}', ha='center',
             fontsize=7.5, color='#6d7480')
    return _save_fig(fig, cfg, outdir, '00_title_card')


def plot_overview(stats, cfg, outdir):
    V = stats['values']
    fig, axs = _new_fig(cfg, 2, 2, (12.5, 8.5))
    _hist_panel(axs[0], V.get('score_notes'), 'Notes per file', 'notes', _C1, logx=True)
    _hist_panel(axs[1], V.get('run_time.total'), 'Play time per file', 'minutes',
                _C2, transform='minutes', logx=True)
    _hist_panel(axs[2], V.get('all_events'), 'MIDI events per file', 'events', _C3, logx=True)
    sd = stats.get('scatter2d') or {}
    counts = sd.get('counts')
    if counts and sum(sum(r) for r in counts) > 0:
        xe, ye = np.asarray(sd['x_edges']), np.asarray(sd['y_edges'])
        C = np.asarray(counts, dtype=float)
        im = axs[3].pcolormesh(xe, ye, np.log1p(C.T), cmap=cfg.cmap, shading='auto')
        axs[3].set_xscale('log')
        axs[3].set_yscale('log')
        cb = fig.colorbar(im, ax=axs[3], shrink=0.85)
        cb.set_label('log(1 + files)')
    else:
        _no_data(axs[3], 'Notes vs play time')
    axs[3].set_title('Notes vs play time (file density)')
    axs[3].set_xlabel('play time, ms (log)')
    axs[3].set_ylabel('notes (log)')
    _finish(fig, cfg, f'{cfg.branding} — Dataset Overview',
            f'{stats["n_files"]:,} feature dicts analyzed')
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    return _save_fig(fig, cfg, outdir, '01_overview')


def plot_quality(stats, cfg, outdir):
    V = stats['values']
    cm_t = V.get('clean_midi.total', {}).get('sum', 0.0)
    cm_c = V.get('clean_midi.clean', {}).get('sum', 0.0)
    drums = sum(v for k, v in (stats['counters'].get('pitches_patches_counts')
                               or {}).items()
                if isinstance(k, tuple) and len(k) == 2 and k[1] == 128)
    if not cm_t:
        return None
    other = max(0.0, cm_t - cm_c - drums)
    fig, axs = _new_fig(cfg, 2, 2, (12.5, 8.8))

    axs[0].pie([cm_c, drums, other],
               labels=[f'melodic (clean)\n{human_big(cm_c)}',
                       f'drums\n{human_big(drums)}',
                       f'other\n{human_big(other)}'],
               colors=['#00b894', '#fdcb6e', '#636e72'],
               autopct='%1.1f%%', startangle=100,
               wedgeprops=dict(width=0.42, edgecolor='white'), textprops={'fontsize': 9})
    axs[0].set_title(f'Note composition — {human_int(cm_t)} notes')

    _hist_panel(axs[1], V.get('ratio.clean'),
                'Melodic (clean) share per file',
                '% of notes on melodic (clean) instruments', _C4, logx=False)

    _hist_panel(axs[2], V.get('bad_chords'), 'Chords fixed per file (bad chords)',
                'chords (log x)', _C1, logx=True)

    rates, rlabels = [], []
    for key, lab in (('ratio.dupe_removed', 'Duplicate pitches removed'),
                     ('ratio.bad_durs', 'Bad durations'),
                     ('ratio.aligned', 'Dominant dtime share')):
        st = V.get(key)
        if st:
            rates.append(st['mean'])
            rlabels.append(lab)
    if rates:
        colors = [_cmap(cfg.cmap)(x) for x in np.linspace(0.15, 0.8, len(rates))]
        y = np.arange(len(rates))
        axs[3].barh(y, rates, color=colors, edgecolor='white')
        axs[3].set_yticks(y)
        axs[3].set_yticklabels(rlabels, fontsize=9)
        axs[3].invert_yaxis()
        for yi, v in zip(y, rates):
            axs[3].text(v + 0.8, yi, f'{v:.1f}%', va='center', fontsize=9, color='#2f3542')
        axs[3].set_xlim(0, max(105, max(rates) * 1.25))
        axs[3].set_xlabel('mean % per file')
        axs[3].set_title('Fix & alignment rates (mean per file)')
    else:
        _no_data(axs[3], 'Fix rates')

    fig.text(0.5, 0.014, '"Clean" = melodic instruments per TMIDIX CLEAN_INSTRUMENTS — '
                         'lead & bass patches (piano, organ, guitar, strings, ensemble,',
             ha='center', fontsize=8, color='#7f8c8d')
    fig.text(0.5, 0.003, 'brass, reed, pipe, synth lead, folk) — as opposed to ethnic, '
                         'percussive & SFX patches. Drums counted separately.',
             ha='center', fontsize=7.5, color='#9aa0a6')
    _finish(fig, cfg, f'{cfg.branding} — Quality & Integrity')
    fig.tight_layout(rect=[0, 0.045, 1, 0.92])
    return _save_fig(fig, cfg, outdir, '02_quality')


def plot_vocabulary(stats, cfg, outdir):
    fc = stats['counters'].get('features_counts')
    if not fc:
        return None
    cats = defaultdict(Counter)
    for tok, v in fc.items():
        cats[token_category(tok)][tok] += v
    order = [c for c in ('delta_time', 'duration', 'pitch', 'velocity',
                         'patch', 'channel', 'chord') if cats.get(c)]
    if not order:
        return None
    avail = {'delta_time': 127, 'duration': 127, 'pitch': 127, 'velocity': 127,
             'patch': 129, 'channel': 16, 'chord': 321}
    totals = [sum(cats[c].values()) for c in order]
    distinct = [len(cats[c]) for c in order]
    fig, axs = _new_fig(cfg, 1, 2, (13, 5.4))
    colors = _cmap(cfg.cmap)(np.linspace(0.1, 0.9, len(order)))
    y = np.arange(len(order))
    axs[0].barh(y, totals, color=colors, edgecolor='white')
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(order, fontsize=9)
    axs[0].invert_yaxis()
    axs[0].set_xscale('log')
    for yi, v in zip(y, totals):
        axs[0].text(v * 1.25, yi, human_big(v), va='center', fontsize=8.5)
    axs[0].set_xlim(right=max(totals) * 8)
    axs[0].set_xlabel('occurrences (log)')
    axs[0].set_title('Token occurrences by category')
    axs[1].barh(y, [avail[c] for c in order], color='#dfe6e9', edgecolor='white',
                label='available in vocab')
    axs[1].barh(y, distinct, color=colors, edgecolor='white', label='distinct values used')
    axs[1].set_yticks(y)
    axs[1].set_yticklabels(order, fontsize=9)
    axs[1].invert_yaxis()
    for yi, (d, a) in enumerate(zip(distinct, order)):
        axs[1].text(avail[a] * 1.05, yi, f'{d}/{avail[a]}',
                    va='center', fontsize=8.5, color='#2f3542')
    axs[1].set_xlim(0, max(avail[c] for c in order) * 1.25)
    axs[1].set_xlabel('distinct tokens')
    axs[1].set_title('Vocabulary coverage (used / available)')
    axs[1].legend(frameon=False, loc='lower right')
    fig.text(0.5, 0.005, 'Each note emits 5 tokens (duration · pitch · velocity · patch · '
             'channel); each chord step adds a delta-time; multi-note chords add a chord token.',
             ha='center', fontsize=8, color='#7f8c8d')
    _finish(fig, cfg, f'{cfg.branding} — Feature-Token Vocabulary')
    fig.tight_layout(rect=[0, 0.04, 1, 0.9])
    return _save_fig(fig, cfg, outdir, '03_vocabulary')


def plot_pitches(stats, cfg, outdir):
    """Pitch profile (melodic), percussion hits (GM kit map), pitch classes, combos."""
    mel = melodic_pitch_profile(stats)
    perc = percussion_profile(stats)
    pp = stats['counters'].get('pitches_patches_counts') or {}
    if not mel and not perc:
        return None
    fig, axs = _new_fig(cfg, 2, 2, (13.5, 9.6))

    if mel:
        ps = sorted(mel)
        vs = [mel[p] for p in ps]
        axs[0].bar(ps, vs, width=0.9, edgecolor='none',
                   color=_cmap(cfg.cmap)(np.linspace(0.15, 0.9, len(ps))))
        lo, hi = max(0, min(ps) - 2), min(127, max(ps) + 2)
        axs[0].set_xlim(lo, hi)
        ticks = [p for p in range(lo, hi + 1) if p % 12 == 0]
        axs[0].set_xticks(ticks)
        axs[0].set_xticklabels([note_name(p) for p in ticks], fontsize=8)
        axs[0].set_title('Pitch profile — melodic instruments')
        axs[0].set_ylabel('notes')
    else:
        _no_data(axs[0], 'Pitch profile (melodic)')

    if perc:
        items = perc.most_common(min(cfg.top_n, len(perc)))[::-1]
        y = np.arange(len(items))
        vs2 = [v for _, v in items]
        axs[1].barh(y, vs2, edgecolor='white',
                    color=_cmap(cfg.cmap)(np.linspace(0.1, 0.9, len(items))))
        axs[1].set_yticks(y)
        axs[1].set_yticklabels([f'{drum_name(k)} ({k})' for k, _ in items], fontsize=8)
        for yi, v in zip(y, vs2):
            axs[1].text(v, yi, ' ' + human_big(v), va='center', fontsize=7.5, color='#2f3542')
        axs[1].set_xlim(right=max(vs2) * 1.2)
        axs[1].set_title(f'Top {len(items)} percussion hits (drum channel, GM kit map)')
        axs[1].set_xlabel('hits')
    else:
        _no_data(axs[1], 'Percussion hits')

    if mel:
        agg = Counter()
        for p, v in mel.items():
            agg[p % 12] += v
        axs[2].bar(range(12), [agg.get(x, 0) for x in range(12)], edgecolor='white',
                   color=_cmap(cfg.cmap)(np.linspace(0.1, 0.9, 12)))
        axs[2].set_xticks(range(12))
        axs[2].set_xticklabels(NOTE_NAMES, fontsize=8.5)
        axs[2].set_title('Pitch classes — melodic instruments')
        axs[2].set_ylabel('notes')
    else:
        _no_data(axs[2], 'Pitch classes')

    combos = [(k, v) for k, v in pp.items() if isinstance(k, tuple) and len(k) == 2]
    if combos:
        combos = sorted(combos, key=lambda t: -t[1])[:cfg.top_n][::-1]
        labels = [f'{pitch_label(k[0], k[1])} @ {patch_label(k[1])}' for k, _ in combos]
        vals = [v for _, v in combos]
        axs[3].barh(range(len(vals)), vals, edgecolor='white',
                    color=_cmap(cfg.cmap)(np.linspace(0.1, 0.9, len(vals))))
        axs[3].set_yticks(range(len(vals)))
        axs[3].set_yticklabels([_trunc(x, 40) for x in labels], fontsize=7.5)
        for yi, v in enumerate(vals):
            axs[3].text(v, yi, ' ' + human_big(v), va='center', fontsize=7.5, color='#2f3542')
        axs[3].set_xlim(right=max(vals) * 1.2)
        axs[3].set_title(f'Top {len(vals)} pitch–instrument combinations')
        axs[3].set_xlabel('notes')
    else:
        _no_data(axs[3], 'Pitch–instrument combinations')

    _finish(fig, cfg, f'{cfg.branding} — Pitches & Percussion')
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    return _save_fig(fig, cfg, outdir, '04_pitches')


def plot_instruments(stats, cfg, outdir):
    pp = stats['counters'].get('pitches_patches_counts')
    by_patch = Counter()
    if pp:
        for k, v in pp.items():
            if isinstance(k, tuple) and len(k) == 2:
                by_patch[k[1]] += v
    if not by_patch:
        return None
    fig, axs = _new_fig(cfg, 1, 2, (13.5, max(5.5, 0.32 * cfg.top_n + 2)))
    items = by_patch.most_common(cfg.top_n)[::-1]
    y = np.arange(len(items))
    vs = [v for _, v in items]
    colors = _cmap(cfg.cmap)(np.linspace(0.1, 0.9, len(items)))
    axs[0].barh(y, vs, color=colors, edgecolor='white')
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([_trunc(patch_label(k), 32) for k, _ in items], fontsize=8.5)
    for yi, v in zip(y, vs):
        axs[0].text(v, yi, ' ' + human_big(v), va='center', fontsize=8, color='#2f3542')
    axs[0].set_xlim(right=max(vs) * 1.2)
    axs[0].set_xlabel('notes')
    axs[0].set_title(f'Top {len(items)} instruments (GM patches)')

    fam = Counter()
    for p, v in by_patch.items():
        fam[family_name(p)] += v
    fitems = fam.most_common()[::-1]
    y2 = np.arange(len(fitems))
    vs2 = [v for _, v in fitems]
    colors2 = _cmap(cfg.cmap)(np.linspace(0.1, 0.9, len(fitems)))
    axs[1].barh(y2, vs2, color=colors2, edgecolor='white')
    axs[1].set_yticks(y2)
    axs[1].set_yticklabels([k for k, _ in fitems], fontsize=8.5)
    for yi, v in zip(y2, vs2):
        axs[1].text(v, yi, ' ' + human_big(v), va='center', fontsize=8, color='#2f3542')
    axs[1].set_xlim(right=max(vs2) * 1.2)
    axs[1].set_xlabel('notes')
    axs[1].set_title('Instrument families (all 16 GM groups + drums)')
    _finish(fig, cfg, f'{cfg.branding} — Instrumentation')
    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    return _save_fig(fig, cfg, outdir, '05_instruments')


def plot_heatmap(stats, cfg, outdir):
    """Square pitch × patch heatmap — full 128×129 grid as a dense log-color field
    (zeros rendered at the colormap floor, like the original, no gray gaps)."""
    pp = stats['counters'].get('pitches_patches_counts')
    if not pp:
        return None
    M = np.zeros((128, 129))
    for k, v in pp.items():
        if isinstance(k, tuple) and len(k) == 2 and 0 <= k[0] < 128 and 0 <= k[1] <= 128:
            M[k[0], k[1]] += v
    if M.sum() <= 0:
        return None
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_axes([0.11, 0.11, 0.72, 0.74])
    im = ax.imshow(np.log1p(M), aspect='equal', origin='lower',
                   cmap=_cmap(cfg.cmap), interpolation='nearest')
    for p in range(12, 128, 12):                     # subtle octave guide lines
        ax.axhline(p - 0.5, color='white', alpha=0.08, lw=0.6)
    ax.set_xticks(range(0, 129, 8))
    ax.set_xticklabels([str(i) + ('*' if i == 128 else '') for i in range(0, 129, 8)],
                       fontsize=7.5)
    ax.set_yticks(range(12, 128, 12))
    ax.set_yticklabels([note_name(p) for p in range(12, 128, 12)], fontsize=7.5)
    ax.set_xlabel('GM patch number   (* = drums — GM percussion note map on the y-axis)')
    ax.set_ylabel('MIDI pitch')
    ax.grid(False)
    ax.set_title(f'{cfg.branding} — Pitch × Instrument density (log color)',
                 fontsize=12, fontweight='bold', pad=14)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('log(1 + note count)')
    fig.text(0.11, 0.045, cfg.branding, fontsize=7.5, color='#9aa0a6')
    fig.text(0.83, 0.045, cfg.attribution, fontsize=7.5, color='#9aa0a6', ha='right')
    return _save_fig(fig, cfg, outdir, '06_pitch_x_instrument_heatmap')


def plot_timing(stats, cfg, outdir):
    fc = stats['counters'].get('features_counts') or {}

    def tok_agg(lo, hi, base):
        agg = Counter()
        for tok, v in fc.items():
            if isinstance(tok, int) and lo <= tok <= hi:
                agg[tok - base] += v
        return agg

    dt, dur, vel = tok_agg(1, 127, 0), tok_agg(129, 255, 128), tok_agg(385, 511, 384)
    if not dt and not dur and not vel:
        return None
    fig, axs = _new_fig(cfg, 2, 2, (12.5, 8.2))
    for ax, agg, title, xlabel, color in (
            (axs[0], dt, 'Inter-chord delta-time', 'delta-time (dataset units; 1 unit ≈ 32 ms)', _C4),
            (axs[1], dur, 'Note durations', 'duration (dataset units; 1 unit ≈ 32 ms)', _C2),
            (axs[3], vel, 'Note velocities', 'velocity (1–127)', _C3)):
        if agg:
            xs = np.array(sorted(agg))
            ys = np.array([agg[x] for x in xs])
            ax.bar(xs, ys, color=color, edgecolor='none')
            ax.set_yscale('log')
            ax.margins(x=0.01)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('count (log)')
        else:
            _no_data(ax, title)
    _hist_panel(axs[2], stats['values'].get('aligned.dtime_ms'),
                'Dominant delta-time per file (true ms)', 'ms', _C1, logx=True)
    _finish(fig, cfg, f'{cfg.branding} — Timing & Expression')
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    return _save_fig(fig, cfg, outdir, '07_timing')


def plot_chords(stats, cfg, outdir):
    fc = stats['counters'].get('features_counts') or {}
    ch = Counter({k: v for k, v in fc.items() if isinstance(k, int) and 657 <= k <= 977})
    if not ch:
        return None
    items = ch.most_common(cfg.top_n)[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.32 * len(items) + 1.8)))
    y = np.arange(len(items))
    vs = [v for _, v in items]
    colors = _cmap(cfg.cmap)(np.linspace(0.1, 0.9, len(items)))
    ax.barh(y, vs, color=colors, edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels([f'#{k - 657} {chord_name(k - 657)}' for k, _ in items], fontsize=8)
    for yi, v in zip(y, vs):
        ax.text(v, yi, ' ' + human_big(v), va='center', fontsize=8, color='#2f3542')
    ax.set_xlim(right=max(vs) * 1.18)
    ax.set_xlabel('occurrences')
    ax.set_title(f'Top {len(items)} chord tokens (chordified tones-chords)', fontsize=12,
                 fontweight='bold')
    _finish(fig, cfg, f'{cfg.branding} — Harmony')
    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    return _save_fig(fig, cfg, outdir, '08_chords')


def plot_cover_galaxy(stats, cfg, outdir):
    """Eye-candy cover #1: square pitch-class galaxy rose on deep space."""
    import matplotlib.patheffects as pe
    prof = Counter()
    for p, v in melodic_pitch_profile(stats).items():
        prof[p % 12] += v
    rng = random.Random(7)
    fig = plt.figure(figsize=(9, 9))
    fig.patch.set_facecolor(_DARK)
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88], polar=True)
    ax.set_facecolor(_DARK)
    total = sum(prof.values())
    if total:
        vals = [prof.get(i, 0) for i in range(12)]
        vmax = max(vals) or 1
        theta = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi / 24
        radii = 0.15 + 0.85 * np.array(vals) / vmax
        colors = _cmap(cfg.cmap)(np.linspace(0.1, 0.95, 12))
        ax.bar(theta, radii, width=2 * np.pi / 12 * 0.86, bottom=0.12, color=colors,
               edgecolor=_DARK, linewidth=1.2, alpha=0.95)
        for t, r, v in zip(theta, radii, vals):
            if v:
                ax.text(t, r + 0.10, f'{100 * v / total:.0f}%', ha='center', va='center',
                        fontsize=8, color='#e8e8ee', clip_on=False)
    ax.set_ylim(0, 1.18)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax.set_xticklabels(NOTE_NAMES, fontsize=10, color='white')
    ax.set_yticks([])
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    for _ in range(140):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0, 1)
        ax.scatter(a, r, s=rng.uniform(1, 8), color='white',
                   alpha=rng.uniform(0.05, 0.4), lw=0)
    notes = stats['values'].get('score_notes', {}).get('sum', 0)
    ms = stats['values'].get('run_time.total', {}).get('sum', 0)
    stroke = [pe.withStroke(linewidth=4, foreground=_DARK)]
    ax.text(math.pi / 2, -0.28, human_big(notes), ha='center', va='center',
            fontsize=22, fontweight='bold', color='white', clip_on=False,
            path_effects=stroke)
    ax.text(math.pi / 2, -0.46, 'NOTES', ha='center', va='center', fontsize=8.5,
            color='#c8cdd6', clip_on=False, path_effects=stroke)
    fig.text(0.5, 0.955, cfg.branding.upper(), ha='center', fontsize=20,
             fontweight='bold', color='white')
    fig.text(0.5, 0.922, 'PITCH-CLASS GALAXY · MELODIC NOTES', ha='center',
             fontsize=9, color='#c8cdd6')
    fig.text(0.5, 0.045, f'{human_big(stats["n_files"])} files · {human_duration_long(ms)}',
             ha='center', fontsize=9, color='#e8e8ee')
    fig.text(0.5, 0.022, cfg.attribution, ha='center', fontsize=7.5, color='#9aa0a6')
    return _save_fig(fig, cfg, outdir, '09_cover_galaxy')


def _fancy_bbox(*a, **kw):
    from matplotlib.patches import FancyBboxPatch
    return FancyBboxPatch(*a, **kw)


def plot_cover_gradient(stats, cfg, outdir):
    """Eye-candy cover #2: square gradient stat-card with key dataset facts."""
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    N = 512
    xs = np.linspace(0, 1, N)
    X, Y = np.meshgrid(xs, xs)
    ax.imshow((X + Y) / 2, extent=(0, 1, 0, 1), origin='lower', cmap=cfg.cmap, aspect='auto')

    def chip(x, y, w, h, label, value):
        value = str(value)
        # auto-fit font size so long values (e.g. 'Acoustic Grand Piano')
        # never overflow the chip
        fs = 15.0
        inner_pts = max(0.1, w - 0.05) * 9 * 72      # axes units → points (9in fig)
        max_chars = inner_pts / (0.60 * fs)          # ~0.60·fs pt per bold char
        if len(value) > max_chars:
            fs = max(9.0, fs * max_chars / len(value))
        ax.add_patch(_fancy_bbox((x, y), w, h,
                                 boxstyle='round,pad=0.012,rounding_size=0.02',
                                 transform=ax.transAxes, fc='#000000',
                                 alpha=0.30, ec='white', lw=0.8))
        ax.text(x + 0.022, y + h - 0.033, label, fontsize=8.5, color='#e8e8ee',
                va='top', transform=ax.transAxes)
        ax.text(x + 0.022, y + 0.035, value, fontsize=fs, fontweight='bold',
                color='white', va='bottom', transform=ax.transAxes)

    V = stats['values']
    notes = V.get('score_notes', {}).get('sum', 0)
    ms = V.get('run_time.total', {}).get('sum', 0)
    dens = V.get('notes_per_sec', {}).get('mean', 0)
    by_patch = Counter()
    for k, v in (stats['counters'].get('pitches_patches_counts') or {}).items():
        if isinstance(k, tuple) and len(k) == 2:
            by_patch[k[1]] += v
    top_inst = patch_name(by_patch.most_common(1)[0][0]) if by_patch else '—'
    mel = melodic_pitch_profile(stats)
    top_pitch = note_name(mel.most_common(1)[0][0]) if mel else '—'

    chip(0.08, 0.60, 0.38, 0.16, 'FEATURE DICTS', human_big(stats['n_files']))
    chip(0.54, 0.60, 0.38, 0.16, 'TOTAL NOTES', human_big(notes))
    chip(0.08, 0.40, 0.38, 0.16, 'TOTAL PLAY TIME', human_duration_long(ms))
    chip(0.54, 0.40, 0.38, 0.16, 'MEAN DENSITY', f'{dens:,.1f} notes/s')
    chip(0.08, 0.20, 0.38, 0.16, 'TOP INSTRUMENT', _trunc(top_inst, 30))
    chip(0.54, 0.20, 0.38, 0.16, 'TOP PITCH', top_pitch)

    fig.text(0.08, 0.86, cfg.branding, fontsize=30, fontweight='bold', color='white')
    fig.text(0.08, 0.815, 'DATASET COVER · FEATURE STATISTICS', fontsize=10, color='#f0f0f5')
    fig.text(0.08, 0.085, cfg.attribution, fontsize=9, color='#f0f0f5')
    fig.text(0.92, 0.085, f'generated {stats["generated"]}', fontsize=8,
             color='#e8e8ee', ha='right')
    return _save_fig(fig, cfg, outdir, '10_cover_gradient')


_PLOTS = [('00_title_card', plot_title_card), ('01_overview', plot_overview),
          ('02_quality', plot_quality), ('03_vocabulary', plot_vocabulary),
          ('04_pitches', plot_pitches), ('05_instruments', plot_instruments),
          ('06_pitch_x_instrument_heatmap', plot_heatmap), ('07_timing', plot_timing),
          ('08_chords', plot_chords), ('09_cover_galaxy', plot_cover_galaxy),
          ('10_cover_gradient', plot_cover_gradient)]


def build_all_plots(stats, cfg):
    _setup_style()
    outdir = Path(cfg.output_dir) / 'plots'
    saved = []
    _log('Rendering plots…', 1, cfg)
    for name, fn in _progress(_PLOTS, cfg, 'Plotting', len(_PLOTS)):
        try:
            p = fn(stats, cfg, outdir)
            if p:
                saved.append(p)
        except Exception as e:
            _log(f'! plot {name} failed: {e}', 0, cfg)
            if cfg.verbosity >= 3:
                traceback.print_exc()
    return saved


# =============================================================== demo data
_DEMO_DRUM_COMMON = [36, 38, 38, 42, 42, 46, 45, 49, 51, 55, 54, 56]


def make_demo_dicts(n_files=120, seed=2026):
    rng = random.Random(seed)
    out = {}
    for i in range(n_files):
        uid = hashlib.md5(f'galaxy-demo-{i}-{rng.randrange(2 ** 32)}'.encode()).hexdigest()
        n_notes = rng.randint(150, 6000)
        n_chords = max(1, n_notes // rng.randint(2, 6))
        patches = rng.sample(range(128), rng.randint(1, 6))
        if rng.random() < 0.45:
            patches.append(128)
        mel_patches = [p for p in patches if p != 128] or [0]
        chans = sorted(rng.sample(range(16), rng.randint(1, min(4, len(patches)))))
        fc, pp = Counter(), Counter()
        for _ in range(n_notes):
            pat = rng.choice(patches)
            if pat == 128:                        # drums: use GM percussion notes
                ptc = rng.choice(_DEMO_DRUM_COMMON if rng.random() < 0.8
                                 else sorted(GM_DRUMS))
            else:
                ptc = max(21, min(108, int(rng.gauss(60, 18))))
            fc[129 + rng.randint(1, 60)] += 1
            fc[257 + ptc] += 1
            fc[385 + rng.randint(40, 112)] += 1
            fc[512 + pat] += 1
            fc[641 + rng.choice(chans)] += 1
            pp[(ptc, pat)] += 1
        for _ in range(rng.randint(50, 600)):
            fc[rng.randint(1, 127)] += 1
        for _ in range(rng.randint(10, n_chords)):
            fc[657 + rng.randrange(321)] += 1
        fc[978] = rng.randint(0, max(1, n_chords // 50))
        fc[979] = max(1, n_chords)
        n_text = rng.randint(1, 30) if rng.random() < 0.35 else 0
        n_lyric = rng.randint(1, 200) if rng.random() < 0.15 else 0
        bad = rng.randint(0, 12)
        clean = int(n_notes * rng.uniform(0.55, 0.98))
        dupes = rng.randint(0, 40) if rng.random() < 0.3 else 0
        rt = int(n_chords * rng.choice([4, 8, 10, 16, 20, 32, 64]) * rng.uniform(50, 90))
        out[uid] = {
            'midi_path': f'demo/song_{i:04d}.mid',
            'aligned': {'dtime_ms': rng.choice([4, 5, 8, 10, 16, 20, 32, 100, 200]),
                        'aligned': int(n_chords * rng.uniform(0.55, 0.98)), 'total': n_chords},
            'all_chords_good': rng.random() < 0.42,
            'all_events': int(n_notes * rng.uniform(1.02, 1.35)) + n_text + n_lyric,
            'bad_durs': {'bad': bad, 'zero': 0, 'total': n_notes,
                         'counts': {rng.randint(128, 1500): 1 for _ in range(min(bad, 8))}},
            'clean_midi': {'clean': clean, 'total': n_notes},
            'dupe_pitches': {'deduped': n_notes - dupes, 'total': n_notes},
            'features_counts': dict(fc),
            'karaoke': {'text': n_text, 'lyric': n_lyric},
            'lyric_events': n_lyric,
            'mono_mels': {rng.choice(mel_patches): rng.randint(20, 500)
                          for _ in range(rng.randint(1, 4))},
            'other_events': rng.randint(0, 300),
            'patch_change_events': rng.randint(0, 20),
            'pitches_patches_counts': dict(pp),
            'run_time': {'total': rt, 'last_time': max(0, rt - rng.randint(0, 5000))},
            'score_chords': n_chords,
            'score_notes': n_notes,
            'text_events': n_text,
            'text_lyric_latin': rng.choice([True, True, None]) if (n_text or n_lyric) else None,
            'tracks': rng.randint(1, 8),
        }
    return out


# ========================================================= high-level API
def print_compact_summary(stats, cfg, paths, elapsed):
    if cfg.verbosity <= 0:
        return
    notes = stats['values'].get('score_notes', {}).get('sum', 0)
    ms = stats['values'].get('run_time.total', {}).get('sum', 0)
    nplots = len(paths.get('plots', []))
    print('─' * REPORT_WIDTH)
    print(f'✓ {cfg.branding} — statistics complete')
    print(f'  Dicts: {stats["n_files"]:,}   Notes: {human_big(notes)}   '
          f'Play time: {human_duration_long(ms)}   Plots: {nplots}')
    if stats.get('capped_runtime') or stats.get('capped_density'):
        print(f'  Outliers capped: {stats.get("capped_runtime", 0):,} runtime · '
              f'{stats.get("capped_density", 0):,} density '
              f'(see report §2 for details)')
    for k in ('report', 'json'):
        if k in paths:
            print(f'  {k + ":":<8} {paths[k]}')
    if nplots:
        print(f'  plots:   {Path(paths["plots"][0]).parent}  ({nplots} files)')
    print(f'  Done in {elapsed:,.1f} s — {cfg.attribution}')
    print('─' * REPORT_WIDTH)


def analyze(dicts=None, inputs=None, cfg=None, t0=None):
    """Full pipeline. `dicts` may be a LIST of feature dicts (10M+ OK, streaming,
    constant-memory), a dict-of-dicts, or any indexable sequence / iterator.
    Items may be bare dicts or {md5_hash: feature_dict} wrappers (auto-detected)."""
    cfg = cfg or Config()
    t0 = t0 or time.time()
    if dicts is None and not inputs:
        raise ValueError('Provide `dicts` (list/dict of feature dicts) or `inputs` (paths).')

    if dicts is not None:
        stats = aggregate_stats(dicts, cfg)
    else:
        stats = aggregate_stats(collect_feature_dicts(inputs, cfg), cfg)

    paths, plots = {}, []
    if cfg.save_plots:
        if HAVE_MPL and HAVE_NUMPY:
            try:
                plots = build_all_plots(stats, cfg)
            except Exception as e:
                _log(f'! plotting failed: {e}', 0, cfg)
                if cfg.verbosity >= 3:
                    traceback.print_exc()
        else:
            _log('! matplotlib/numpy unavailable — plots skipped.', 0, cfg)
    paths['plots'] = plots

    report = build_text_report(stats, cfg)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    if cfg.save_text_report:
        p = Path(cfg.output_dir) / 'report.txt'
        p.write_text(report, encoding='utf-8')
        paths['report'] = str(p)
    if cfg.save_json:
        p = Path(cfg.output_dir) / 'stats_summary.json'
        save_json_summary(stats, p)
        paths['json'] = str(p)

    if cfg.verbosity >= 2:
        print(report)
    print_compact_summary(stats, cfg, paths, time.time() - t0)
    return stats, paths


# Convenience alias for the programmatic (large list) workflow.
analyze_features = analyze


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='galaxy_midi_stats',
        description=f'{DEFAULT_BRANDING} — MIDI feature-dataset statistics, reports & plots.',
        epilog=DEFAULT_ATTRIBUTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('inputs', nargs='*',
                    help='*.json files/dirs of feature dicts, and/or *.mid/*.midi (needs TMIDIX)')
    ap.add_argument('--demo', action='store_true', help='run on built-in synthetic demo data')
    ap.add_argument('-o', '--outdir', default='gmidi_stats', help='output directory')
    ap.add_argument('-v', '--verbosity', type=int, default=1, choices=(0, 1, 2, 3),
                    help='0 quiet · 1 normal · 2 verbose (full report) · 3 debug')
    ap.add_argument('--no-progress', action='store_true', help='disable progress bars')
    ap.add_argument('-j', '--workers', type=int, default=-1,
                    help='parallel workers (-1 = all cores, 1 = sequential)')
    ap.add_argument('--backend', default='loky',
                    choices=('loky', 'threading', 'multiprocessing', 'sequential'))
    ap.add_argument('--batch', type=int, default=20000, help='feature dicts per parallel batch')
    ap.add_argument('--top', type=int, default=15, help='top-N rows in tables/plots')
    ap.add_argument('--max-runtime-ms', type=int, default=86_400_000,
                    help='clip per-file run_time to this many ms (outlier guard; 0 = off)')
    ap.add_argument('--max-density', type=float, default=1000.0,
                    help='clip per-file notes/sec density (outlier guard; 0 = off)')
    ap.add_argument('--no-plots', action='store_true', help='skip plot rendering')
    ap.add_argument('--show', action='store_true', help='also display plots interactively')
    ap.add_argument('--no-json', action='store_true', help='skip JSON summary')
    ap.add_argument('--no-report-file', action='store_true', help='skip report.txt')
    ap.add_argument('--branding', default=DEFAULT_BRANDING)
    ap.add_argument('--attribution', default=DEFAULT_ATTRIBUTION)
    ap.add_argument('--cmap', default='plasma', help='matplotlib colormap for plots')
    args = ap.parse_args(argv)

    if not args.inputs and not args.demo:
        ap.error('no inputs given (pass JSON/MIDI paths or use --demo)')

    cfg = Config(output_dir=args.outdir, verbosity=args.verbosity,
                 progress=not args.no_progress, workers=args.workers,
                 backend=args.backend, batch_size=args.batch, top_n=args.top,
                 max_runtime_ms=args.max_runtime_ms, max_density_nps=args.max_density,
                 save_plots=not args.no_plots, show_plots=args.show,
                 save_json=not args.no_json, save_text_report=not args.no_report_file,
                 branding=args.branding, attribution=args.attribution, cmap=args.cmap)

    if cfg.show_plots and HAVE_MPL and matplotlib.get_backend().lower() == 'agg':
        for bk in ('TkAgg', 'QtAgg', 'MacOSX'):
            try:
                matplotlib.use(bk)
                break
            except Exception:
                continue

    if args.demo:
        _log('Generating demo dataset…', 1, cfg)
        analyze(dicts=make_demo_dicts(), cfg=cfg, t0=time.time())
    else:
        analyze(inputs=args.inputs, cfg=cfg, t0=time.time())


if __name__ == '__main__':
    main()
#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#	galaxymidi Helpers Python Module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2026
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
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
'''

print('=' * 70)
print('Loading galaxymidi helpers module...')
print('Please wait...')
print('=' * 70)

__version__ = '1.0.0'

print('galaxymidi helpers module version', __version__)
print('=' * 70)

###################################################################################

import os
import shutil
import subprocess
import time

import hashlib

from .TMIDIX import midi2score, score2midi

from typing import List, Dict

###################################################################################

def sort_aligned_lists(midi_names: list[str],
                       midi_sequences: list[list[int]],
                       reverse: bool = False
                       ) -> None:
    
    """
    Sort two aligned lists in place based on the length of each MIDI sequence.

    Parameters
    ----------
    midi_names : list[str]
        List of MIDI file names. Must be aligned with `midi_sequences`.
    midi_sequences : list[list[int]]
        List of MIDI sequences (or any list-like objects). Each element's length
        is used as the sorting key. Must be aligned with `midi_names`.
    reverse : bool, optional
        If True, sort in descending order (longest first). If False (default),
        sort in ascending order (shortest first).

    Notes
    -----
    - Sorting is **stable** because Python's `sorted()` is stable.
    - Sorting is **in-place**: both lists are mutated and remain aligned.
    - The function does not return anything; it modifies the input lists directly.

    Returns
    -------
    None
    """
    
    idx = sorted(range(len(midi_sequences)),
                 key=lambda i: len(midi_sequences[i]),
                 reverse=reverse)

    midi_names[:] = [midi_names[i] for i in idx]
    midi_sequences[:] = [midi_sequences[i] for i in idx]
    
###################################################################################

def get_normalized_midi_md5_hash(midi_file: str) -> Dict:
    
    """
    Helper function which computes normalized MD5 hash for any MIDI file
    
    Returns
    -------
    Dictionary with MIDI file name, original MD5 hash and normalized MD5 hash
    {'midi_name', 'original_md5', 'normalized_md5'}
    """

    bfn = os.path.basename(midi_file)
    fn = os.path.splitext(bfn)[0]
    
    midi_data = open(midi_file, 'rb').read()
    
    old_md5 = hashlib.md5(midi_data).hexdigest()
    
    score = midi2score(midi_data, do_not_check_MIDI_signature=True)
        
    norm_midi = score2midi(score)
    
    new_md5 = hashlib.md5(norm_midi).hexdigest()
    
    output_dic = {'midi_name': fn,
                  'original_md5': old_md5,
                  'normalized_md5': new_md5
                 }

    return output_dic

###################################################################################

def normalize_midi_file(midi_file: str,
                        output_dir: str = '',
                        output_file_name: str = ''
                        ) -> str:
    
    """
    Helper function which normalizes any MIDI file and writes it to disk
    
    Returns
    -------
    Path string to a written normalized MIDI file
    """

    if not output_file_name:
        output_file_name = os.path.basename(midi_file)

    if not output_dir:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)

    midi_path = os.path.join(output_dir, output_file_name)

    if os.path.exists(midi_path):
        fn = os.path.splitext(output_file_name)[0]
        output_file_name = f'{fn}_normalized.mid'
        midi_path = os.path.join(output_dir, output_file_name)
    
    midi_data = open(midi_file, 'rb').read()
    
    score = midi2score(midi_data, do_not_check_MIDI_signature=True)
        
    norm_midi = score2midi(score)

    with open(midi_path, 'wb') as fi:
        fi.write(norm_midi)

    return midi_path

###################################################################################

def is_installed(pkg: str) -> bool:
    """Return True if package is already installed (dpkg-query)."""
    try:
        subprocess.run(
            ["dpkg-query", "-W", "-f=${Status}", pkg],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        # dpkg-query returns "install ok installed" on success
        out = subprocess.run(["dpkg-query", "-W", "-f=${Status}", pkg],
                             stdout=subprocess.PIPE, text=True).stdout.strip()
        return "installed" in out
    except subprocess.CalledProcessError:
        return False

###################################################################################

def _run_apt_get(args, timeout):
    base = ["apt-get", "-y", "-o", "Dpkg::Options::=--force-confdef", "-o", "Dpkg::Options::=--force-confold"]
    cmd = base + args
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)

###################################################################################

def install_apt_package(pkg: str,
                        update: bool = True,
                        timeout: int = 600,
                        require_root: bool = True,
                        use_python_apt: bool = False
                        ) -> Dict:
    
    """
    Install an apt package idempotently.
    - pkg: package name (e.g., 'fluidsynth')
    - update: run apt-get update first
    - timeout: seconds for apt operations
    - require_root: if True, will prefix with sudo when not root (may prompt)
    - use_python_apt: try python-apt API first if True
    
    Returns
    -------
    Status dict {'status', 'package'}
    """
    
    if is_installed(pkg):
        return {"status": "already_installed", "package": pkg}

    # Optionally try python-apt (requires python-apt installed and running as root)
    if use_python_apt:
        try:
            import apt
            cache = apt.Cache()
            cache.update()
            cache.open(None)
            if pkg in cache:
                pkg_obj = cache[pkg]
                if not pkg_obj.is_installed:
                    pkg_obj.mark_install()
                    cache.commit()
                return {"status": "installed_via_python_apt", "package": pkg}
        except Exception:
            # fall through to subprocess fallback
            pass

    # Build command environment
    prefix = []
    if require_root and os.geteuid() != 0:
        if shutil.which("sudo"):
            prefix = ["sudo"]
        else:
            raise PermissionError("Root privileges required and sudo not available.")

    # Optionally update
    if update:
        tries = 5
        for attempt in range(tries):
            try:
                subprocess.run(prefix + ["apt-get", "update"], check=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                break
            except subprocess.CalledProcessError as e:
                if attempt + 1 == tries:
                    raise
                time.sleep(2 ** attempt)

    # Install with retry for transient locks
    tries = 6
    for attempt in range(tries):
        try:
            subprocess.run(prefix + ["apt-get", "-y", "install", pkg], check=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if is_installed(pkg):
                return {"status": "installed", "package": pkg}
            else:
                raise RuntimeError("apt-get reported success but package not found installed.")
        except subprocess.CalledProcessError as e:
            # common cause: dpkg lock; backoff and retry
            if "Could not get lock" in e.stderr or "dpkg was interrupted" in e.stderr:
                time.sleep(2 ** attempt)
                continue
            raise
            
    raise RuntimeError("Failed to install package after retries.")

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the galaxymidi Helpers Python Module
###################################################################################
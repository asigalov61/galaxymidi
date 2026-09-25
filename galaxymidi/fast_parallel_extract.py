#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#	Fast Parallel Extract Python Module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2025
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
###################################################################################
###################################################################################
#
#   Copyright 2025 Project Los Angeles / Tegridy Code
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
#   Critical packages
#
#   !sudo apt update -y
#   !sudo apt install -y p7zip-full
#   !sudo apt install -y pigz
#
###################################################################################
###################################################################################
#
#   Basic use example
#
#   import fast_parallel_extract
#
#   fast_parallel_extract.fast_parallel_extract()
#
###################################################################################
'''

###################################################################################

import os
import shutil
import subprocess
import time

###################################################################################

def fast_parallel_extract(archive_path='./Galaxy-MIDI-Dataset/Galaxy-MIDI-Dataset-CC-BY-NC-SA.tar.gz', 
                          output_dir='./', 
                          pigz_procs=256
                         ):

    """Fast, parallel extraction of a tar.gz archive using `tar` piped through `pigz`.
    
    This helper extracts a compressed tar archive to `output_dir` while leveraging
    `pigz` (parallel gzip) for multi-core decompression. It constructs a `tar`
    command that uses `pigz -p <n>` as the decompressor and runs it via
    `subprocess.run`, suppressing command output. The function measures and prints
    elapsed time and performs basic validation of inputs and environment.
    
    Parameters
    ----------
    archive_path : str, optional
        Path to the tar.gz archive to extract. Default:
        `'./Galaxy-MIDI-Dataset/Galaxy-MIDI-Dataset-CC-BY-NC-SA.tar.gz'`.
    output_dir : str, optional
        Destination directory where archive contents will be extracted. The
        directory is created if it does not already exist. Default:
        `'./'`.
    pigz_procs : int, optional
        Number of parallel `pigz` worker processes to request via `pigz -p`.
        Higher values increase decompression parallelism but may saturate CPU or
        I/O. Choose a value appropriate for your machine. Default: `64`.
    
    Returns
    -------
    None
        The function performs extraction as a side effect and prints progress and
        timing information to stdout. It does not return the path or any status.
    
    Raises
    ------
    FileNotFoundError
        If `archive_path` does not exist.
    EnvironmentError
        If the `pigz` executable is not found in the system `PATH`.
    OSError, subprocess.CalledProcessError
        If `tar` or `pigz` fail to run or the subprocess encounters an OS-level
        error. Note that the current implementation does not explicitly check the
        subprocess return code; failures may not raise unless the OS reports an
        error to `subprocess.run`.
    
    Notes
    -----
    - This implementation is POSIX-oriented: it writes subprocess output to
      `"/dev/null"`. On non-POSIX systems (for example, Windows) `"/dev/null"` may
      not exist; use `os.devnull` or adapt the code for cross-platform compatibility.
    - The function checks for `pigz` using `shutil.which` and raises an
      `EnvironmentError` if it is not available. Ensure `pigz` is installed and
      accessible in the PATH before calling this function.
    - `pigz_procs` should be tuned to the number of available CPU cores and the
      characteristics of the storage device. Very large values can cause CPU
      contention and reduce throughput, especially on HDDs or network filesystems.
    - The `tar` command is constructed as:
      ```
      tar -I "pigz -p <pigz_procs>" -xvf <archive_path> -C <output_dir>
      ```
      which instructs `tar` to use the specified `pigz` invocation for
      decompression.
    - The function suppresses both stdout and stderr of the extraction command by
      redirecting them to `"/dev/null"`. If you need to debug extraction failures,
      remove or modify the redirection so output is visible or capture it via
      `subprocess.run(..., capture_output=True)`.
    
    Security
    --------
    - The function executes an external command constructed from user-provided
      paths. Ensure `archive_path` and `output_dir` are trusted and not attacker-
      controlled to avoid command injection or unexpected behavior. Prefer passing
      arguments as a list (as done here) and avoid shell=True.
    
    """
    
    print('=' * 70)
    print('Extracting...')

    start_time = time.time()

    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"The archive file '{archive_path}' does not exist.")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if shutil.which("pigz") is None:
        raise EnvironmentError("The 'pigz' package is not installed or is not found in the system's PATH.")

    command = [
        "tar",
        "-I", f"pigz -p {pigz_procs}",
        "-xvf", archive_path,
        "-C", output_dir
    ]

    with open("/dev/null", "w") as devnull:
        subprocess.run(command, stdout=devnull, stderr=subprocess.STDOUT)

    end_time = time.time()
    execution_time = end_time - start_time

    print('Done!')
    print('=' * 70)
    print(f"Extraction took {execution_time / 60} minutes")
    print('=' * 70)

###################################################################################
# This is the end of the Fast Parallel Extract Python Module
###################################################################################
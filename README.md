# Slepian focusing computational toolkit

Copyright (c) 2012-2024 Kornel JAHN (kornel.jahn@gmail.com)

:warning: **This code base is not being developed actively, only maintained  just to save it from code rot** :warning:

:warning: **Preparation of some simple examples still pending**

This repository contains code related to my PhD thesis entitled "The vector Slepian theory of high numerical aperture focusing" (2017), available online at [BME MDA, the Budapest University of Technology and Economics Digital Archives](https://repozitorium.omikk.bme.hu/items/60f9d49e-3c48-4397-b04e-9cdd5e0884d3).

## Repository structure

The repository defines a Python package `slepianfocusing`, installable using
e.g. `pip`, and also includes some runnable usage examples in the `examples`
directory.

## Installation

### Prerequisites

- Python 3.11 or later; if not available, install it
  - *on Windows*:
    - from [the official Python download site](https://www.python.org/downloads),
    - via the command-line package manager `winget` as
      ```
      winget install -e --id Python.Python.311
      ```
    - from the Microsoft Store\
  - *on Linux*: using the package manager of the Linux distribution

### Installation of the `slepianfocusing` Python package

1. Clone this repository locally
2. Create and activate a virtual environment (venv):
    ```
    python -m venv env
    ```

   *NOTE:* in the above command, it is assumed that the Python interpreter can simply be called as `python`. This command might not be present on the actual OS or might not call the right Python version. In this case, replace it by e.g. `python3.11.exe` (on Windows) or `python3` or `python3.11` (on Linux).

3. Activate the venv:
  - *on Windows*:
    - *Windows Command Prompt* (a.k.a. `cmd.exe`):
      ```
      env\Scripts\activate.bat
      ```
    - *Windows PowerShell*:
      ```
      env\Scripts\Activate.ps1
      ```
  - *on Linux* (assuming a POSIX-compatible shell):
    ```sh
    . env/bin/activate
    ```

    *NOTE:* if activation in PowerShell fails with the message that *"[...] cannot be loaded because running scripts is disabled on this system"*, execute the following command in PowerShell:
    ```
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```

    **After activating the venv, the Python interpreter can simple be called using the `python` command!**

4. (Recommended) Upgrade `pip`:
   ```
   python -m pip install --upgrade pip
   ```

5. Local ("editable") installation of `slepianfocusing` into the venv that follows source code changes, with all development dependencies included:
   ```
   pip install -e .[devel]
   ```

## Running the unit tests

Assuming the `slepianfocusing` package has been installed with its development
dependencies, activate the venv and run
```
pytest tests
```

## Running the examples (under construction!)

After activating the vnev, examples inside the `examples` directory can be run
simply as, for instance,
```
python examples/TODO.py
```

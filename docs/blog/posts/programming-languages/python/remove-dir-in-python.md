---
date: 2023-11-17
authors: [hermann-web]
comments: true
description: >
  File handling is crucial in Python programming. This documentation provides insights into managing directories, specifically targeting empty and non-empty directories using `os` and `pathlib` modules.
categories:
  - programming
  - python
  - file-handling
title: "Removing Directories in Python"
---

## Remove Empty Directories in Python

To remove an **empty directory**, you can use `os.rmdir()` with os and `Path.rmdir()` with pathlib.

=== ":octicons-file-code-16: `index.py`"

    ```python
    import os

    directory_path = '/path/to/empty_directory'

    try:
        os.rmdir(directory_path)
        print(f"The directory '{directory_path}' has been successfully removed.")
    except OSError as e:
        print(f"Error: {directory_path} : {e.strerror}")
    ```

=== ":octicons-file-code-16: `index.py`"

    ```python
    from pathlib import Path

    directory_path = Path('/path/to/empty_directory')

    try:
        directory_path.rmdir()
        print(f"The directory '{directory_path}' has been successfully removed.")
    except OSError as e:
        print(f"Error: {directory_path} : {e.strerror}")
    ```

<!-- more -->

## Remove Non-Empty Directories in Python

For directories that contain files or other directories, use `shutil.rmtree()` to remove them along with their contents.

=== ":octicons-file-code-16: `index.py`"

    ```python
    import shutil

    directory_path = '/path/to/non_empty_directory'

    try:
        shutil.rmtree(directory_path)
        print(f"The directory '{directory_path}' and its contents have been successfully removed.")
    except OSError as e:
        print(f"Error: {directory_path} : {e.strerror}")
    ```

=== ":octicons-file-code-16: `index.py`"

    ```python
    from pathlib import Path
    import shutil

    directory_path = Path('/path/to/non_empty_directory')

    try:
        shutil.rmtree(directory_path)
        print(f"The directory '{directory_path}' and its contents have been successfully removed.")
    except OSError as e:
        print(f"Error: {directory_path} : {e.strerror}")
    ```
Ensure to confirm the directory paths before removal operations to prevent accidental deletion of important data. When using shutil.rmtree(), exercise caution as it permanently deletes directories and their contents. The choice between os and pathlib can depend on your preference for object-oriented or procedural-style programming when handling paths and directories in Python.

## relating links

- [Pathlib Tutorial: Transitioning to Simplified File and Directory Handling in Python](./pathlib-transition-tutorial.md)

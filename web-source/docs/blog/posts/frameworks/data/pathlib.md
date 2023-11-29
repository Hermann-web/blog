---
date: 2023-11-28
authors: [hermann-web]
description: >
  Learn to simplify file and directory handling in Python using pathlib.
categories:
  - frameworks
  - dev
  - python
  - file-handling
links:
  - setup/setting-up-a-blog.md
  - plugins/blog.md
title: "pathlib Tutorial: Simplifying File and Directory Handling in Python"
---

# pathlib Tutorial: Simplifying File and Directory Handling in Python

## Introduction

__Are you still using `import os` for file handling after 2020 ? Use `pathlib` instead !__

In this tutorial, we'll explore the powerful `pathlib` module in Python, which offers an elegant and intuitive way to work with file paths, directories, and file system operations.

### Why `pathlib`?

Traditional file handling in Python often involves using the `os` module or string manipulations for path operations. However, `pathlib` offers a more object-oriented approach, simplifying path manipulations and providing a more readable and platform-independent solution.

Let's dive into various functionalities of `pathlib` step by step.

## Getting Started: Initializing Paths

First, let's begin by initializing a `Path` object.

```python
from pathlib import Path

# Initialize a Path object
path = Path("path/to/file/or/dir")
```

<!-- more -->

## Joining Paths

One of the strengths of `pathlib` is the ability to join paths easily.

```python
# Joining paths
path = path / "rrrr" / "rrrr.txt"
```

## Path Operations

### Retrieving Absolute Path and Parent Directory

You can easily obtain the absolute path and the parent directory using `resolve()` and `parent`.

```python
# Get the absolute path
abspath = path.resolve()

# Get the parent directory
parent = path.parent
```

## Checking Path Type and Existence

### File and Directory Checks

`pathlib` simplifies checking if a path points to a file or directory.

```python
# Check if it's a file
is_file = path.is_file()

# Check if it's a directory
is_dir = path.is_dir()
```

### Existence Check

Verify if the path exists.

```python
# Check if path exists
path_exists = path.exists()
```

## File System Manipulations

### Creating and Deleting Files

`pathlib` provides straightforward methods for file creation and deletion.

```python
# Create a file (if it doesn't exist)
path.touch()

# Delete a file
path.unlink()
```

### Creating and Deleting Directories

Similarly, it simplifies directory creation and deletion.

```python
# Create a folder/directory
path.mkdir()

# Delete a folder (if it's empty)
path.rmdir()
```

## Conclusion

Congratulations! You've now gained a foundational understanding of `pathlib` and its capabilities for file and directory handling in Python. Experiment further by combining these methods and explore additional functionalities for your file manipulation needs.

Stay curious and keep exploring to harness the full potential of `pathlib` in your Python projects!

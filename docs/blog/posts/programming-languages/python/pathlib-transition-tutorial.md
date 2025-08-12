---
date: 2023-12-20
authors: [hermann-web]
comments: true
description: >
  Learn how to transition from command line or 'os' module to Python's 'pathlib' for simplified file and directory handling.
categories:
  - programming
  - python
  - file-handling
  - tools-comparison
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: "Pathlib Tutorial: Transitioning to Simplified File and Directory Handling in Python"
---

# Pathlib Tutorial: Transitioning to Simplified File and Directory Handling in Python

## Introduction

__Are you still using `import os` for file handling after 2020 ? Use `pathlib` instead !__

If you're moving away from command line operations or 'os' module to Python's `pathlib`, you're at the right place.

Well, in this tutorial, we'll dive into the powerful `pathlib` module in Python. It offers a clean transition for users accustomed to CLI or 'os' for file and directory handling, providing an elegant and intuitive approach.

<!-- more -->

## Key Considerations for Choosing os or pathlib

While `os` and `pathlib` both handle file and directory operations, they differ significantly in their approach:

=== ":octicons-file-code-16: `os`"

    - **Procedural:** Primarily functions for path operations.
    - **Built-in:** A part of Python's standard library.
    - **String-based:** Paths represented as strings.

=== ":octicons-file-code-16: `pathlib`"

    - **Object-oriented:** Paths as Path objects.
    - **Introduced in Python 3.4:** Not available in older Python versions.
    - **Enhanced functionality:** Offers extensive methods for path manipulation.
    - **Cross-platform:** Works well across different operating systems.

## Making the Transition

- __Preference for Object-Oriented Approach:__ `pathlib` provides a more intuitive and readable experience.
- __Compatibility and Legacy Code:__ `os` may be necessary for older Python versions or existing code.
- __Specific Functionality:__ Some advanced operations might be easier with one module over the other.
- __Project Style and Conventions:__ Consider the overall project style and best practices.

## Best Practices for Smooth Transition

- __Consistency:__ Maintain a uniform approach within a project for easier maintenance.
- __Descriptive Naming:__ Use clear variable names for paths to enhance readability.
- __Error Handling:__ Implement robust error handling for potential issues.
- __Thorough Testing:__ Ensure correctness in file and directory operations.

### Why Transition to `pathlib`?

Traditionally, file handling in Python often relied on the command line or the 'os' module. However, `pathlib` introduces an object-oriented paradigm, offering an intuitive and platform-independent solution.

The transition to `pathlib` allows for:

- Simplified Path Representation: Paths as Path objects offer enhanced readability and functionality.
- Streamlined Syntax: Code becomes concise and more understandable using `pathlib`'s methods.
- Expanded Methodology: `pathlib` covers common path operations comprehensively.

Let's explore the functionalities of `pathlib` step by step.

!!! example "Usage Examples"

    ```python
    from pathlib import Path

    # Get the current working directory
    cwd = Path.cwd()

    # Create a new directory
    new_dir = Path("my_new_directory")
    new_dir.mkdir()

    # Create a nested directory structure
    nested_dir = Path("data/processed/results")
    nested_dir.mkdir(parents=True, exist_ok=True)  # Create all parent directories if needed

    # Check if a file exists
    file_path = Path("my_file.txt")
    if filepath.exists():
        print("The file exists!")

    # Read the contents of a file
    text = filepath.read_text()

    # Write content to a file
    filepath.write_text("New content for the file")

    # Remove an empty directory
    empty_dir = Path("empty_dir")
    empty_dir.rmdir()

    # Remove a non-empty directory and its contents
    non_empty_dir = Path("non_empty_dir")
    shutil.rmtree(non_empty_dir)
    ```

## Equivalence os vs pathlib vs cli for Common Operations

This table provides a comparison between Python's `os` module and the `pathlib` library operations alongside their Linux command equivalents for file and directory manipulation, information retrieval, traversal, file input/output (I/O), and path validation. It offers a comprehensive reference for developers familiar with Python who want to understand corresponding operations in Linux command line interfaces. The table is categorized by groups, making it easy to find specific functionalities and their corresponding commands in Python and Linux.

??? note "More on Table description"

    This table is a comprehensive guide detailing various file and directory operations along with path manipulation using Python's `os` and `pathlib` modules, alongside their Linux command equivalents.

    - **Path Manipulation:** Operations for obtaining the current working directory, joining paths, and checking path existence are provided.

    - **File and Directory Information:** It includes functions to get file/directory size, list directory contents, and retrieve file-related timestamps like creation, modification, and access times.

    - **File and Directory Operations:** This section covers creating and removing directories, renaming files/directories, copying files/directories, as well as commands for creating and deleting files directly.

    - **Path Information and Validation:** Functions to check if a path points to a file or directory, checking if a path is absolute, and extracting file extensions are included.

    - **Path Traversal and Exploration:** Operations to iterate through files matching a specific pattern, resolve absolute paths, and extract the parent directory or file/directory name are outlined.

    - **File I/O:** Covers reading, writing, and appending contents to a file.

    - **Path Accessibility:** How to check for path accessibility (read, write, execute permissions) using Python's `os` module alongside Linux command equivalents is explained.

    Each operation is represented under its relevant group, detailing the equivalent Pythonic approach using `os` or `pathlib`, alongside their corresponding Linux command line alternatives. This table serves as a quick reference for performing common file system-related tasks using Python and Linux commands.

| Group                            | Operation                                   | os                                       | pathlib                                                       | Linux Command Equivalent         |
|----------------------------------|---------------------------------------------|------------------------------------------|---------------------------------------------------------------|---------------------------------|
| Path Manipulation                | Get current working directory               | `os.getcwd()`                            | `Path.cwd()`                                                  | `pwd`                           |
|                                  | Join paths                                  | `os.path.join('/path', 'to', 'join')`    | `Path('/path') / 'to' / 'join'`                                 | `joinpath`                      |
|                                  | Check path existence                        | `os.path.exists('/path')`                | `Path('/path').exists()`                                       | `test -e /path`                 |
| File and Directory Information  | Get file/directory size                     | `os.path.getsize('/path')`               | `Path('/path').stat().st_size`                                 | `du -b /path`                   |
|                                  | List directory contents                     | `os.listdir('/path')`                   | `[item.name for item in Path('/path').iterdir()]`             | `ls /path`                      |
|                                  | Get file creation time                      | `os.path.getctime('/path')`              | `Path('/path').stat().st_ctime`                                | `stat -c %W /path`              |
|                                  | Get file last modification time              | `os.path.getmtime('/path')`              | `Path('/path').stat().st_mtime`                                | `stat -c %Y /path`              |
|                                  | Get file last access time                   | `os.path.getatime('/path')`              | `Path('/path').stat().st_atime`                                | `stat -c %X /path`              |
| File and Directory Operations    | Create a directory                          | `os.makedirs('/path')`                   | `Path('/path').mkdir()` (for single directory), `Path('/path').mkdir(parents=True)` (for nested directories) | `mkdir /path`                    |
|                                  | Remove an empty directory                   | `os.rmdir('/path')`                      | `Path('/path').rmdir()`                                        | `rmdir /path`                   |
|                                  | Remove a non-empty directory                | `shutil.rmtree('/path')`                 | `shutil.rmtree(Path('/path'))`                                | `rm -r /path`                   |
|                                  | Rename a file/directory                    | `os.rename('path/to/source', 'path/to/dest')`  | `Path('path/to/source').rename('path/to/dest')`              | `mv path/to/source path/to/dest` |
|                                  | Copy a file                                | `shutil.copy('/source/file', '/destination/file')` | `Path('/source/file').replace('/destination/file')`         | `cp /source/file /destination/file` |
|                                  | Copy a directory                           | `shutil.copytree('/source/dir', '/destination/dir')` | `shutil.copytree('/source/dir', '/destination/dir')`         | `cp -r /source/dir /destination/dir` |
|                                  | Create a file                              | `open('/file/path', 'w').close()`        | `Path('/file/path').touch()`                                   | `touch /file/path`              |
|                                  | Remove a file                              | `os.remove('/file/path')`                | `Path('/file/path').unlink()`                                  | `rm /file/path`                 |
| Path Information and Validation  | Check if the path is a file                 | `os.path.isfile('/path')`                | `Path('/path').is_file()`                                      | `test -f /path`                 |
|                                  | Check if the path is a directory            | `os.path.isdir('/path')`                 | `Path('/path').is_dir()`                                       | `test -d /path`                 |
|                                  | Check if the path is absolute               | `os.path.isabs('/path')`                 | `Path('/path').is_absolute()`                                  | `readlink -f /path`             |
|                                  | Get the file extension                      | `os.path.splitext('/path')[1]`           | `Path('/path').suffix`                                         | `echo /path | grep -o -P '\.\K.*'` |
| Path Traversal and Exploration    | Iterate through files matching a pattern    | `glob.glob('/path')`                     | `Path('/path').glob()` or `Path('/path').rglob()` for recursive search | `find /path -name pattern`       |
|                                  | Resolve the absolute path                   | `os.path.abspath('/path')`              | `Path('/path').resolve()`                                      | `realpath /path`                |
|                                  | Get the parent directory                    | `os.path.dirname('/path')`               | `Path('/path').parent`                                         | `dirname /path`                 |
|                                  | Get the file/directory name                 | `os.path.basename('/path')`              | `Path('/path').name`                                           | `basename /path`                |
| File I/O                          | Read the contents of a file                 | `open('/file/path'', 'r').read()`      | `Path('/file/path'').read_text()`                              | `cat /file/path`                |
|                                  | Write contents to a file                    | `open('/file/path'', 'w').write(content)` | `Path('/file/path'').write_text(content)`                     | `echo content > /file/path`     |
|                                  | Append contents to a file                   | `open('/file/path', 'a').write(content)` | `Path('/file/path'').open('a').write(content)`                | `echo content >> /file/path`    |
| Path Accessibility                | Check path accessibility                    | `os.access()`                            | Use `Path` methods in combination with `os.access()` or `os.stat()` | `test -rwx /path`               |

!!! note "Remember !"
    - Always handle potential errors with `try-except` blocks for file and directory operations.
    - Test your code thoroughly to ensure correctness and handle edge cases.
    - Embrace the object-oriented nature of pathlib for a more intuitive and readable approach to file handling in Python.

## Conclusion

Congratulations! You've now gained a foundational understanding of `pathlib` and its capabilities for file and directory handling in Python. Experiment further by combining these methods and explore additional functionalities for your file manipulation needs.

Stay curious and keep exploring to harness the full potential of `pathlib` in your Python projects!

## Related Posts

- [Cheat on Python Package Managers](../../../posts/programming-languages/python/package-managers-in-python.md)
- [Removing Directories in Python](../../../posts/programming-languages/python/remove-dir-in-python.md)

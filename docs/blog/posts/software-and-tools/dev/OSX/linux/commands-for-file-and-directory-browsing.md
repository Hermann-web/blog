---
date: 2023-12-30
authors: [hermann-web]
comments: true
description: >
  Master essential Linux commands for efficient file and directory operations. Explore `wc`, `du`, `grep`, `find`, ... and their combinations for effective file management.
title: "Mastering Essential Linux Commands: Your Path to File and Directory Mastery"
categories:
  - devops
  - OSX
  - linux
  - command-line
  - file-handling
  - deployment
---

# Mastering Essential Linux Commands: Your Path to File and Directory Mastery

## Introduction

This documentation aims to offer a comprehensive understanding of essential commands and techniques for file and directory management in a Linux environment. Mastering these commands is crucial for efficient navigation, manipulation, and analysis of files and directories.

We'll embark on a journey by delving into the foundational usage of key commands like `wc`, `du`, `grep`, and `find`, uncovering their individual functionalities. Additionally, we'll explore how these commands can be combined using powerful methods such as pipes (`|`), `-exec {} \;`, or `-exec {} +`, unlocking their synergistic potential.

Moreover, to solidify your understanding, real-life examples showcasing practical applications will be demonstrated. 

<!-- more -->

The hands-on experience gained through testing and implementing these commands will be pivotal in comprehending their nuanced usage and unleashing their practical utility.

Let the learning begin !

## 1. Basic Commands Overview

### wc (Word Count)

The `wc` command is used to count lines, words, and characters in files.

- **Counting lines in a file:**
  ```bash
  wc -l file.txt
  ```
  This command displays the number of lines in `file.txt`.

### du (Disk Usage)

The `du` command estimates file and directory space usage.

- **Getting the size of a directory:**
  ```bash
  du -h /path/to/directory
  ```
  This command provides the disk usage of the specified directory (`/path/to/directory`) in a human-readable format (`-h`).

### grep (Global Regular Expression Print)

The `grep` command searches for patterns in files.

- **Searching for lines containing a pattern in a file:**
  ```bash
  grep "pattern" file.txt
  ```
  This command displays lines in `file.txt` that contain the specified `pattern`.

### `head` and `tail`

The `head` command displays the beginning of a file, while `tail` shows the end.

- **Viewing the first few lines of a file with `head`:**
  ```bash
  head file.txt
  ```
  This command displays the first few lines of `file.txt`.

- **Displaying a specific number of lines at the beginning of a file with `head -n`:**
  ```bash
  head -n 10 file.txt
  ```
  This command displays the first 10 lines of `file.txt`. You can replace `10` with any number to view a different quantity of lines.

- **Viewing the last few lines of a file with `tail`:**
  ```bash
  tail file.txt
  ```
  This command shows the last few lines of `file.txt`.

- **Displaying a specific number of lines at the end of a file with `tail -n`:**
  ```bash
  tail -n 15 file.txt
  ```
  This command shows the last 15 lines of `file.txt`. Similarly, you can adjust `15` to any desired number to see a different quantity of lines.

Using `-n` with `head` or `tail` allows you to precisely control the number of lines displayed from the beginning or end of a file.

### `less` and `more`

Both `less` and `more` are used to view text files in a paginated manner.

- **Viewing a file with `less`:**
  ```bash
  less file.txt
  ```
  `less` allows you to navigate through the file interactively.

- **Viewing a file with `more`:**
  ```bash
  more file.txt
  ```
  `more` displays the file content page by page, but it has more limited navigation options compared to `less`.

These commands provide different ways to view file contents, either scrolling through the entire file or just a section at a time.

## 2. `find` command

### basic search

The `find` command searches for files and directories based on various criteria.

- **Finding files by name:**
  ```bash
  find /path/to/search -name "filename.txt"
  ```
  Searches for `filename.txt` in `/path/to/search` and its subdirectories.

- **Finding files matching pattern:**
  ```bash
  find /path/to/search -name "req*.txt"
  ```
  Searches ... in `/path/to/search` and its subdirectories.

- **Finding files by type:**
  ```bash
  find /path/to/search -type f
  ```
  Finds all files in `/path/to/search` and its subdirectories.

    !!! info "`-type f` search files when `-type d` is for directories"

- **Find and delete files:**
  ```bash
  find /path/to/search -name "file_to_delete.txt" -delete
  ```
  This command finds a file named `file_to_delete.txt` and deletes it.

### find: Enhanced Searching with more options
- **Find files by size:**
  ```bash
  find /path/to/search -size +10M
  ```
  This command finds files larger than 10 megabytes in the specified directory.

- **Find files modified within a time range:**
  ```bash
  find /path/to/search -mtime -7
  ```
  This command finds files modified within the last 7 days in the specified directory.


### fast conclusion
These commands offer different functionalities:
- `wc` counts lines, words, or characters in a file.
- `du` estimates disk usage for files and directories.
- `grep` searches for patterns in files and prints lines containing the specified pattern.
- the `find` command in Linux is a powerful tool used for searching files and directories based on various criteria.

You can use these commands to perform various operations related to file content, size estimation, and pattern matching within files.


## 3. Combining find with Other Commands
In this section, We explore how to combine the previous commands using pipes (`|`), `-exec {} \;`, or `-exec {} +`:

### Pipes (`|`)

Using pipes to pass the output of one command as input to another.

- **Finding specific files and counting them:**
    ```bash
    find /path/to/search -name "*.txt" | wc -l
    ```
    Finds `.txt` files and counts them using `wc -l`.

### `-exec {} \;`: Find and perform an action on each file

Executing a command for each matched file or directory.

- **Finding files and displaying their sizes:**
    ```bash
    find /path/to/search -type f -exec du -h {} \;
    ```

    !!! info "Displays sizes of files (each file in a different command) found by `find` using `du -h`."

- **Finding files and performing deletion:**
    ```bash
    find /path/to/search -name "file_to_delete.txt" -exec rm {} \;
    ```

    !!! info "Deletes files (each file in a different command) matching the name `file_to_delete.txt`."

- **Finding and searching patterns:**
    ```bash
    find /path/to/search -name "*.txt" -exec grep "pattern" {} \;
    ```

    !!! info "This command finds all `.txt` files in the specified directory and runs `grep` to search for a specific pattern within each of those files."

### `-exec {} +`: Find and perform an action on all files at once

Optimizing efficiency by passing multiple arguments to a command.

- **Finding files and performing deletion:**
    ```bash
    find /path/to/search -name "file_to_delete.txt" -exec rm {} +
    ```

    !!! info "Deletes files (all in one command) matching the name `file_to_delete.txt`"

### fast conclusion
`find` is an incredibly versatile command that can be combined with various flags and options to perform advanced searches based on filenames, types, sizes, modification times, and more. It's a great tool for locating specific files or performing actions on groups of files based on specific criteria.


## 4. Application showcases

### Counting Files in a Folder

To count the number of files in a folder, you can use the following commands:

1. Using `find`:
    ```bash
    find /path/to/folder -maxdepth 1 -type f | wc -l
    ```

    !!! info "More"
        - This command uses `find` to search for files (`-type f`) in the specified folder without going into subdirectories (`-maxdepth 1`). 
        - The output is then piped to `wc -l`, which counts the number of lines, effectively giving you the count of files.

2. Using `ls`:
    ```bash
    ls -l /path/to/folder | grep "^-" | wc -l
    ```

    !!! info "More"
        Here, 
        - `ls -l` lists the contents of the folder with detailed information
        - `grep "^-"` filters out only the lines that represent files (as opposed to directories or other types of items)
        - `wc -l` counts the number of lines, providing the count of files in the folder.


### Counting Files/Folders in a Folder

```bash
find /path/to/folder -maxdepth 1 | wc -l
```
or
```bash
ls -l /path/to/folder | wc -l
```

### Determining the Number of Columns in a CSV File

```bash
head -n 1 input/google-form-data.csv | grep -o "," | wc -l
```

!!! info "This command reads the first line, apply the `,` separator then count"

### Finding `requirements.txt` Files Containing "openpyxl"

- find all requirements.txt files
```bash
find . -name requirements.txt
```

- find all requirements.txt files who contain "openpyxl"
```bash
find . -name requirements.txt -exec grep -l "openpyxl" {} \;
```

### Utilizing `maxdepth` for Search

- find all `.txt` files non recursively 

```bash
find . -maxdepth 1 -type f -name "*.txt"
```

- find ...
```bash
find . -maxdepth 3
```

- save the result
```bash
find . -maxdepth 3 > output.txt
```

### Skipping Certain Paths in a Search

- find all py files but skip venv folders (paths containing venv)
```bash
find . -name "*.py" ! -path "*venv*"
```

- find all py files but skip venv folders and apply yapf on each file
```bash
find . -name "*.py" ! -path "*venv*" -exec yapf --in-place {} \;
```

- find all py files but skip folders(likely env) and apply yapf on each file
```bash
find . -name "*.py" ! -path "*env/Scripts*" -exec yapf --in-place {} \;
```

- search files where the word wrappers is mentionned and avoid some folders
```bash
find . -type f -not -path '*/node_modules/*' -not -path '*env*' -not -name '*_*' -name '*.py' -exec grep -l 'wrappers' {} +
```

### git grep for version controlled files
- search files where the word wrappers is mentionned withing the version controlled files 
```bash
git grep -l "wrapper" -- "*.py"
```

### Using `tail` Command

- display the last 50 lines of a file
```bash
tail -50 cli.log
```

- filter the output of another command
```bash
tail -50 cli.log | grep "/api/"
``` 
This command will display the last 50 lines of the `cli.log` file and filter out only the lines that contain "/api/". This combination of `tail` and `grep` will help you isolate and display the relevant lines.


### search for lines containing the word "black" within `.sh` files

```bash
find /path/to/search -type f -name "*.sh" -exec grep -l "black" {} +
```

??? info "More on find and grep options"
    This command will search for lines containing the word "black" within `.sh` files. The command (`grep`) displays the actual lines containing "black" within the files

    - **grep options: `grep` vs `grep -l`**
        To only show the filenames without the matches, use the command (`grep`) instead of (`grep -l`)

    - **`-exec` option in the `find` command**
        This syntax uses `+` at the end of the `-exec` option. It gathers the file names that match the criteria (`*.sh`) and passes them to `grep` in batches, rather than invoking `grep` once per file. This is generally more efficient, especially when dealing with a large number of files.

        To invoke `grep` individually for each file that matches the criteria (`*.sh`), use instead 
        `find /path/to/search -type f -name "*.sh" -exec grep -l "black" {} \;`: This syntax uses `\;` at the end of the `-exec` option. 

        This method might be less efficient, especially for a large number of files, as it starts a new `grep` process for each file separately.


### search for folders

- search folder by name (ex:all name containing eigen3)
```bash
find -type d -name "*eigen3*"
```

- search for a specific folder like "LAStools/bin" starting from the root directory `/home`. 
```bash
sudo find /home -type d -name "bin" -path "*LAStools*"
```

!!! info "This command searches the entire root directory `/` for directories (`-type d`) named "bin" (`-name "bin"`) that are part of a path containing "LAStools" (`-path "*LAStools*"`)"

!!! warning "Using `sudo` might be necessary to have permission to search directories that your user account doesn't have access to by default."

## Conclusion

These commands, when mastered and strategically combined, offer a robust toolkit for proficiently managing and manipulating files and directories in a Linux environment. By leveraging these commands in tandem, users can perform intricate searches, conduct comprehensive analyses, and execute operations swiftly, significantly enhancing productivity and workflow efficiency.

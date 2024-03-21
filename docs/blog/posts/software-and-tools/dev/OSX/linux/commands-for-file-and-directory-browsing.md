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

We'll embark on a journey by delving into the foundational usage of key commands like `wc`, `du`, `grep`, `awk`, and `find`, uncovering their individual functionalities. Additionally, we'll explore how these commands can be combined using powerful methods such as pipes (`|`), `-exec {} \;`, or `-exec {} +`, unlocking their synergistic potential.

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

??? info "Additional `grep` options:"

    - `-H`: Print the filename for each match when searching in multiple files.
      ```bash
      grep -H "pattern" file1.txt file2.txt
      ```

    - `-l`: Display only the names of files that contain at least one match, instead of showing the matching lines.
      ```bash
      grep -l "pattern" file1.txt file2.txt
      ```

    - `-n`: Display the line numbers along with the matching lines.
      ```bash
      grep -n "pattern" file.txt
      ```

    - `-w`: Match the whole word, ensuring that only entire words are considered.
      ```bash
      grep -w "word" file.txt
      ```

    - `-i`: Perform case-insensitive matching, ignoring differences in case when searching for the pattern.
      ```bash
      grep -i "pattern" file.txt
      ```
    
    - `-B N`: Display N lines before the matching line.
      ```bash
      grep -B 2 "pattern" file.txt
      ```

    - `-A N`: Display N lines after the matching line.
      ```bash
      grep -A 2 "pattern" file.txt
      ```

    These options enhance the functionality of `grep` by providing more context, line numbers, and filename information when searching for patterns in files.

### awk (Aho, Weinberger, and Kernighan)

- **Basic Syntax:**

  ```bash
  awk 'pattern { action }' file.txt
  ```

  - `pattern`: The condition that a line must meet to trigger the action.
  - `action`: The set of commands to be executed when the pattern is matched.

    !!! example "Example"

        ```bash
        awk '/pattern/ { print $1 }' file.txt
        ```
        This command prints the first field of each line in `file.txt` where the pattern is found.

    ??? info "Common Use Cases:"

        1. **Printing Specific Columns:**
          ```bash
          awk '{ print $2, $4 }' file.txt
          ```
          This prints the second and fourth columns of each line in `file.txt`.

        2. **Pattern Matching:**
          ```bash
          awk '/error/ { print $0 }' log.txt
          ```
          Prints lines containing the word "error" from the `log.txt` file.

        3. **Calculations:**
          ```bash
          awk '{ total += $1 } END { print total }' numbers.txt
          ```
          Calculates and prints the sum of the values in the first column of `numbers.txt`.

        4. **Custom Field and Record Separators:**
          ```bash
          awk -F',' '{ print $1 }' data.csv
          ```
          Specifies ',' as the field separator in a CSV file.

    ??? info "Advanced Features:"

        - **Variables:**
          ```bash
          awk '{ total += $1 } END { print "Sum:", total }' numbers.txt
          ```
          Uses the variable `total` to accumulate values.

        - **Built-in Functions:**
          ```bash
          awk '{ print length($0) }' text.txt
          ```
          Prints the length of each line in `text.txt`.

        - **Conditional Statements:**
          ```bash
          awk '{ if ($1 > 10) print $0 }' values.txt
          ```
          Prints lines where the value in the first column is greater than 10.

`awk` is versatile and can be highly customized for various text processing tasks. It's especially useful for working with structured data in files.

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

    !!! info "for case insensitive search, use `-iname` instead of `-name`

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

### Conclusion

These commands offer different functionalities:

- `wc` counts lines, words, or characters in a file.
- `du` estimates disk usage for files and directories.
- `grep` searches for patterns in files and prints lines containing the specified pattern.
- `awk` is a powerful text processing tool for pattern scanning and processing.
- The `find` command in Linux is a powerful tool used for searching files and directories based on various criteria.

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

### `\(` ... `\)`: Grouping Expressions

When using `find` to search for files based on multiple criteria, such as file name patterns, types, or sizes, you may need to combine these criteria using logical operators like `-and`, `-or`, or `-not`. The `\( ... \)` construct allows you to group these expressions together to ensure they are evaluated as a single logical unit.

Grouping multiple expressions together for logical operations.

- **Grouping Expressions in `find`:**

    ```bash
    find /path/to/search \( -name "*.txt" -o -name "*.pdf" \) -size +1M
    ```

    !!! info "Groups the conditions for finding files with either `.txt` or `.pdf` extensions and with a size greater than 1MB."

    Using `\( ... \)` allows for the proper grouping of expressions within a `find` command, ensuring that logical operations are applied correctly.

Overall, `\( ... \)` is a crucial construct in `find` commands for combining multiple search criteria and ensuring their proper evaluation. It helps create more complex search patterns while maintaining clarity and precision in the command syntax.

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

### Using `head` and `tail` Commands

- display the last 50 lines of a file

```bash
tail -50 cli.log
```

- filter the output of another command

```bash
tail -50 cli.log | grep "/api/"
```

This command will display the last 50 lines of the `cli.log` file and filter out only the lines that contain "/api/". This combination of `tail` and `grep` will help you isolate and display the relevant lines.

- install the first lines of `requirements.txt` using `head` and `xargs`

```bash
head -n 18 requirements.txt | xargs -n 1 pip3 install
```

This command will read the first 18 lines of `requirements.txt`, then install each package listed there using `pip3`.

!!! tip "An improvement of this command has been proposed [here](../../../../programming-languages/python/integrating-requirements-with-poetry.md) using `sed` to remove from the requirement file, spaces, comment, empty lines, ..."

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

### Searching for "AAA" in Files

- search for "AAA" in all files

```bash
find . -type f -exec grep "AAA" {} \;
```

- search for "AAA" in all files with file name and line number display

```bash
find . -type f -exec grep -Hn "AAA" {} \;
```

Certainly! Let's integrate the additional commands into the existing explanation:

```bash
### Searching for a keyword in Files from the parent directory

- search for "L337" in all files
```bash
find .. -type f -exec grep -Hn "L337" {} \;
```

- search for "L337" in all files with 5 lines after each match

```bash
find .. -type f -exec grep -Hn "L337" {} \; -A 5
```

This command will find all occurrences of "L337" in files within the parent directory and its subdirectories and display the filename, line number, and the line containing "L337", along with the five lines that follow it.

### Get unique lines across files

- see the unique lines common to all three files without repetitions

```bash
sort file1.txt file2.txt file3.txt | uniq -c | awk '$1 == 3 {print $2}'
```

**Alternative**

```bash
sort file1.txt file2.txt file3.txt | uniq -c | awk '$1 >= 3 {print $2}'
```

- use the precedent list to filter lines from another file (`people`)

```bash
grep -f <(sort file1.txt file2.txt file3.txt | uniq -c | awk '$1 >= 3 {print $2}') ../people
```

This command will first find the unique lines common to all three files, then filter those lines using `grep -f` based on the patterns present in the specified file (in this case, the file containing sorted and unique lines from `file1.txt`, `file2.txt`, and `file3.txt`). Finally, it will display the lines that match in both sets and also contain the term `people`.

### Additional Commands

- extract names of females with the name 'Annabel' from the 'people' file

```bash
cat people | awk '$3 == "F"' | grep 'Annabel' | awk '{print $1, $2}'
```

In the project, this command filter the lines of the file `people` containing the word `Annabel` and where the person is female (3rd fiel == `F`) then use `awk` to print from the filtered file, only the first and second fields. The fields in each line are separated by a space

- search for 'Annabel' in all files and extract the names of females with the name 'Annabel'

```bash
find . -type f -exec grep -w 'Annabel' {} \; -exec awk '$3 == "F" {print $1, $2}' {} \;
```

This command will apply the precedent operation on each file returned by the `find` command

### Grouping Examples

??? example "Finding files with either ".txt" or ".pdf" extensions and with a size greater than 1MB"

    Suppose you want to find files with either ".txt" or ".pdf" extensions and with a size greater than 1MB. You can use `\( ... \)` to group the size condition with the extension conditions:

    ```bash
    find /path/to/search \( -name "*.txt" -o -name "*.pdf" \) -size +1M
    ```
   
    In this command:
   
    - `\( -name "*.txt" -o -name "*.pdf" \)` groups the conditions for finding files with ".txt" or ".pdf" extensions.
    - `-size +1M` specifies the condition for files with a size greater than 1MB.
   
    By grouping the extension conditions together, you ensure that the size condition is applied to both ".txt" and ".pdf" files.

??? example "Finding specific files with specific extensions"

    Suppose you want to find specific files with extensions such as ".sh", ".md", or "Dockerfile" and then search for a particular pattern within them. You can use the following command:

    ```bash
    find /home/ubuntu/Documents/GitHub/ \( -name "*.sh" -o -name "*.md" -o -name "Dockerfile" \) -exec grep -Hn "apt install ./mongodb-database-tools-*.deb &" {} \;
    ```
   
    In this command:
   
    - `\( -name "*.sh" -o -name "*.md" -o -name "Dockerfile" \)` groups the conditions for finding files with the specified extensions.
    - `-exec grep -Hn "apt install ./mongodb-database-tools-*.deb &" {} \;` executes the `grep` command to search for the specified pattern within each matched file.
   
    The `\( ... \)` construct is used to group the `-name` expressions together. This grouping is necessary because the `-o` operator (logical OR) has lower precedence than the implicit logical AND applied to separate `find` expressions. By using `\( ... \)`, you ensure that the logical OR operation is applied correctly within the grouped expressions.
   
    Without the grouping, the command would not function as intended because each `-name` expression would be evaluated separately, potentially leading to unexpected results.

??? example "running a linter script md files from a repo subfolder and the readme file in the main directory"

    To find both `.md` files in the `./docs` directory and `README.md` files in the current directory, you can use the `-o` (OR) operator along with the `-exec` option. Here's how you can do it:

    ```bash
    find . \( -path "./docs" -name "*.md" -o -path "./README.md" \) -exec markdownlint-cli2 --fix {} +
    ```
    
    In this command:
    
    - `.`: Specifies the current directory as the starting point for the `find` command.
    - `\( ... \)`: Groups conditions together.
    - `-path "./docs" -name "*.md"`: Specifies files with the `.md` extension in the `./docs` directory.
    - `-o`: Acts as the logical OR operator.
    - `-path "./README.md"`: Specifies the `README.md` file in the current directory.
    - `-exec markdownlint-cli2 --fix {} +`: Executes the `markdownlint-cli2 --fix` command on the found files. The `{}` is replaced by the found file names.
    
    This command will execute `markdownlint-cli2 --fix` on all `.md` files in the `./docs` directory and `README.md` in the current directory.

## Conclusion

These commands, when mastered and strategically combined, offer a robust toolkit for proficiently managing and manipulating files and directories in a Linux environment. By leveraging these commands in tandem, users can perform intricate searches, conduct comprehensive analyses, and execute operations swiftly, significantly enhancing productivity and workflow efficiency.

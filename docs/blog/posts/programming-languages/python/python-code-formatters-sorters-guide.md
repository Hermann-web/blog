---
date: 2024-02-12
authors: [hermann-web]
comments: true
description: >
  Introduction to Python code formatters and sorters including Black, isort, YAPF, and more, with usage examples and advanced configurations. Learn how to supercharge your CI with these tools for efficient code formatting.
categories:
  - programming
  - python
  - code-quality
  - tools-comparison
title: "Introduction to Python Code Formatters and Sorters"
---

# Introduction to Python Code Formatters and Sorters

## Introduction

__Are you struggling to maintain consistent formatting in your Python code? Do you find yourself spending too much time organizing imports or adjusting code style manually?__

 <div class="float-img-container float-img-right">
  <a href="https://ibb.co/9th9Q2w"><img src="https://i.ibb.co/FVXbRJs/python-code-carbon.png" alt="python-code-carbon" border="0"></a>
</div>

Navigating the landscape of Python code formatters and sorters can be overwhelming, especially for beginners.

This guide serves as your roadmap to mastering Python code formatters and sorters, simplifying the process and providing practical examples for effective code formatting and organization.

<!-- more -->

Whether you're a novice Python developer or an experienced programmer looking to streamline your workflow, this document is tailored to demystify Python code formatters and sorters, offering insights into popular tools like Black, isort, and YAPF. By the end, you'll have a solid understanding of how to leverage these tools to enhance the readability and maintainability of your Python codebase.

## Overview

Python code formatters and sorters are tools designed to improve the readability and maintainability of Python code by enforcing consistent formatting and organization standards. These tools automatically format your code according to predefined rules, saving time and ensuring adherence to best practices. In this document, we'll explore some popular Python code formatters and sorters, including Black, isort, YAPF, and more.

## Black

**Black** is a highly opinionated Python code formatter that reformats your code to follow a consistent style guide. It offers a "one true way" approach to code formatting, ensuring that all code formatted with Black adheres to the same style. Black is known for its simplicity and ability to produce clean, readable code without configuration.

### Installation

You can install Black using pip:

```bash
pip install black
```

### Usage

To format a Python file with Black, simply run:

```bash
black your_file.py
```

??? note "More Options for Black"

    Black provides several options for formatting Python code, including:

    - Format a specific file:
    ```bash
    black myfile.py
    ```

    - Format all Python files in the current directory:
    ```bash
    black *.py
    ```

    - Format all Python files in a directory and its subdirectories:
    ```bash
    black myfolder
    ```

    - Formatting with Black Using a Maximum Line Length of 100 Characters
        ```bash
        black . -l 100
        ```

## isort

**isort** is a Python utility that sorts import statements within your Python code. It organizes import statements alphabetically and groups them according to predefined categories. isort ensures a consistent import order across your codebase, improving readability and maintainability.

### Installation

You can install isort using pip:

```bash
pip install isort
```

### Usage

To sort import statements within a Python file with isort, run:

```bash
isort your_file.py
```

??? note "More Options for isort"

    - Sorting Imports with isort Using the Black Profile

        To sort import statements using isort with the Black profile, run:

        ```bash
        isort --profile black
        ```

## YAPF (Yet Another Python Formatter)

**YAPF** is a Python code formatter developed by Google. It aims to provide a highly configurable code formatting solution while maintaining simplicity and ease of use. YAPF offers a wide range of formatting options, allowing you to customize its behavior according to your project's needs.

### Installation

You can install YAPF using pip:

```bash
pip install yapf
```

### Usage

To apply YAPF to a single file, use the following command:

```bash
yapf -i your_file.py
```

Replace `your_file.py` with the name of the Python file you want to format.

To apply YAPF to multiple files, specify each file separated by spaces:

```bash
yapf -i file1.py file2.py file3.py
```

To apply YAPF recursively to all Python files in a folder, navigate to the folder containing your Python files and run:

```bash
yapf -i ./*.py
```

This command will format all `.py` files in the current directory.

??? note "More Options for YAPF"

    - **Specify Line Length:** You can specify the line length for formatting using the `-l` or `--style` option:

        ```bash
        yapf -i --style="{based_on_style: google, column_limit: 120}" your_file.py
        ```

        This command formats `your_file.py` with a line length limit of 120 characters, based on the Google style guide.

    - **Preserve Existing Formatting:** YAPF allows you to preserve the existing formatting to some extent using the `--style` option:

        ```bash
        yapf -i --style="{based_on_style: pep8, indent_width: 4}" your_file.py
        ```

        This command formats `your_file.py` while preserving the existing indentation width of 4 spaces, based on the PEP 8 style guide.

    - **Dry Run Mode:** You can use the `--diff` option to perform a dry run and see the proposed changes without actually modifying the files:

        ```bash
        yapf --diff your_file.py
        ```

        This command displays the proposed changes to `your_file.py` without modifying the file itself.

    - **Recursive Formatting:** To format Python files in the current directory and all subdirectories, use the `--recursive` flag:

        ```bash
        yapf --in-place --recursive .
        ```

        This command recursively formats all `.py` files starting from the current directory and traversing through all subdirectories.

## Other Python Code Formatters and Sorters

In addition to Black, isort, and YAPF, there are several other Python code formatters and sorters available, each with its own set of features and capabilities. Some notable mentions include:

- __autopep8__: A tool that automatically formats Python code to conform to the PEP 8 style guide.
- __pyfmt__: A code formatter that aims to balance flexibility and readability, offering customizable formatting options.
- __pycodestyle__: A tool that checks Python code against the PEP 8 style guide and provides suggestions for improvements.

## Supercharge Your CI with Formatters

!!! example "Combining isort and Black"

    You can combine isort and Black to format and sort your Python files efficiently:

    ```bash
    #!/bin/bash
    isort .
    black . -l 100
    ```

    This script first sorts the import statements with isort and then formats the code with Black, ensuring a consistent style and organization.

!!! example "Example with find + isort and Black"

    To recursively format and sort Python files in a directory and its subdirectories, while excluding certain directories like 'env' and '.git', you can use the following commands:

    ```bash
    #!/bin/bash
    find . -type f -name "*.py" ! -path "*env/*" ! -path "*.git/*" -exec isort {} +
    find . -type f -name "*.py" ! -path "*env/*" ! -path "*.git/*" -exec black -l 100 {} +
    ```

    These commands will find all Python files in the current directory and its subdirectories, excluding specific directories like 'env' and '.git', and then apply isort and Black with a maximum line length of 100 characters, ensuring clean and well-organized code. Skipping these directories helps avoid unnecessary file processing and ensures a faster and more efficient formatting process.

## Conclusion

Python code formatters and sorters are valuable tools for improving the consistency, readability, and maintainability of Python code. By using tools like Black, isort, YAPF, and others, developers can ensure that their code adheres to established standards and best practices, leading to more efficient and collaborative development processes.

## Related links

- [Some notes on the right formater to use for your project](https://github.com/orsinium/notes/blob/master/notes-python/black.md)

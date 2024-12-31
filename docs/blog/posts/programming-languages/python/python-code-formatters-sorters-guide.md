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
title: "Exploring the Commands and Comparing Python Formatters and Linters: black vs. flake8 vs. isort vs. autopep8 vs. yapf vs. pylint and more"
---

# Exploring the Commands and Comparing Python Formatters and Linters: black vs. flake8 vs. isort vs. autopep8 vs. yapf vs. pylint and more

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

## Key Considerations

### Choosing the Right Tool

- __Code Formatting vs. Code Analysis:__ Distinguish between tools primarily focused on formatting (black, autopep8, yapf, isort) and those focused on analysis (flake8, pylint).
- __Customizability:__ Assess the level of customization offered by each tool to align with project-specific style guidelines and requirements.
- __Integration:__ Consider compatibility and integration with existing development workflows, IDEs, and automation tools.
- __Performance:__ Evaluate the speed and resource consumption of each tool, especially for large codebases.
- __Community and Support:__ Explore the size and activity of the user community, available documentation, and support resources.

### Tools Overview

=== ":arrows_counterclockwise: `isort`"

    - __Import Sorter:__ Isort is a utility for sorting and organizing Python imports within code files.
    - __Automatic Import Sorting:__ Automatically organizes imports alphabetically and groups them by type (standard library, third-party, local imports).
    - __Configuration:__ Offers a variety of configuration options through a `pyproject.toml` or `setup.cfg` file to customize import sorting behavior.
    - __Integration:__ Easily integrates with other tools and formatters such as black to ensure consistent code styling.
    - __Performance:__ Known for its fast execution speed, making it suitable for large codebases.

=== ":hammer_and_wrench: `black`"

    - __Code Formatter:__ Black is an opinionated code formatter that enforces a consistent code style throughout Python projects.
    - __Automatic Formatting:__ Requires no configuration and automatically reformats code to comply with PEP 8 standards.
    - __Uncompromising:__ Focuses on minimizing differences between code styles, favoring simplicity and readability over customization.
    - __Integration:__ Seamlessly integrates with most IDEs, text editors, and CI/CD pipelines.
    - __Community Support:__ Backed by a vibrant community and actively maintained.

=== ":page_with_curl: `yapf`"

    - __Yet Another Python Formatter:__ Yapf is a Python code formatter that aims for readability, consistency, and ease of use.
    - __Configurable Formatting:__ Offers various formatting options and presets to customize code styling according to project preferences.
    - __Integration:__ Supports integration with popular IDEs, text editors, and automation tools for seamless code formatting.
    - __Presets:__ Provides built-in presets for different Python styles, allowing users to quickly apply common formatting configurations.
    - __Community:__ Backed by an active community and ongoing development efforts.

=== ":repeat: `autopep8`"

    - __Automatic Code Formatter:__ Autopep8 automatically formats Python code to conform to the PEP 8 style guide.
    - __Prescriptive Formatting:__ Provides simple and prescriptive formatting rules, focusing on improving code consistency and readability.
    - __Command-Line Tool:__ Can be used as a command-line tool or integrated into text editors and IDEs for automatic code formatting.
    - __Configuration:__ Allows some degree of customization through command-line options and configuration files.
    - __Community:__ Supported by a community of users and actively maintained.

=== ":rocket: `ruff`"

    - __Code Linter and Formatter:__ Ruff is a fast and highly customizable code linter and formatter for Python, combining the functionality of tools like flake8 and black.
    - __Automatic Formatting:__ Ruff can automatically format Python code to comply with PEP 8 standards.
    - __Customizable:__ Ruff offers a variety of configuration options through a `.ruff.toml` or `pyproject.toml` file to customize code analysis and formatting behavior.
    - __Integration:__ Ruff seamlessly integrates with most IDEs, text editors, and CI/CD pipelines.
    - __Community Support:__ Backed by a growing community and actively maintained.

=== ":snowflake: `flake8`"

    - __Code Linter:__ Flake8 is a code analysis tool that checks Python code against coding style (PEP 8) and detects various programming errors.
    - __Modular Architecture:__ Consists of several plugins for code style enforcement, syntax checking, and error detection (e.g., pep8, mccabe, pyflakes).
    - __Customizable:__ Allows configuration through a `.flake8` configuration file to adjust rules and behavior according to project requirements.
    - __Extensibility:__ Supports custom plugins and extensions to enhance functionality and add additional checks.
    - __Usage:__ Typically used as part of CI/CD pipelines or integrated into text editors for real-time feedback.

=== ":eyeglasses: `pylint`"

    - __Code Checker:__ Pylint is a static code analysis tool that checks Python code for errors, potential bugs, and adherence to coding standards.
    - __Extensive Checks:__ Performs a wide range of checks including code style, error detection, code complexity analysis, and more.
    - __Highly Configurable:__ Offers extensive configuration options through a `.pylintrc` file to adjust the level of strictness and enable/disable specific checks.
    - __Integration:__ Integrates with various IDEs, text editors, and CI/CD pipelines for automated code analysis and feedback.
    - __Learning Curve:__ May have a steeper learning curve due to its comprehensive feature set and configuration options.

=== ":mag: `mypy`"

    - __Static Type Checker:__ Mypy is a static type checker for Python, allowing developers to catch type errors during development.
    - __Optional Typing:__ Supports optional typing through type annotations, enabling developers to add type information to their code.
    - __Gradual Typing:__ Allows for gradual typing, allowing developers to incrementally add type annotations to existing codebases.
    - __Integration:__ Seamlessly integrates with most IDEs, text editors, and CI/CD pipelines.
    - __Community Support:__ Backed by a vibrant community and actively maintained.

### Comparison Table

| Feature                        | Type                                          | Focus                                       | Configuration                                | Automatic Formatting                        | Integration                                 | Customization                                   | Speed                                       | Error Detection                             |
|--------------------------------|-----------------------------------------------|---------------------------------------------|----------------------------------------------|---------------------------------------------|---------------------------------------------|-------------------------------------------------|---------------------------------------------|---------------------------------------------|
| [isort](https://github.com/PyCQA/isort)       | Import Sorter                  | Import Sorting                              | Configurable                                 | Yes                                         | Compatible                                  | Customizable (Configuration Files)              | Fast                                        | Limited to Import Errors                    |
| [black](https://github.com/psf/black)         | Formatter                      | Code Formatting                             | Minimal (No Configuration)                   | Yes                                         | Seamless Integration                        | Limited (Follows Strict Rules)                  | Fast                                        | Limited to Formatting Errors                |
| [yapf](https://github.com/google/yapf)        | Formatter                      | Code Formatting                             | Highly Configurable                          | Yes                                         | Integration with IDEs, Text Editors         | Extensive (Configuration Options)               | Moderate                                    | Limited to Formatting Errors                |
| [autopep8](https://github.com/hhatto/autopep8)| Formatter                      | Code Formatting                             | Configurable                                 | Yes                                         | Integration with IDEs, Text Editors         | Limited (Some Configuration Options)            | Moderate                                    | Limited to Formatting Errors                |
| [ruff](https://github.com/astral-sh/ruff)     | Linter and Formatter           | Code Analysis and Formatting                | Highly Configurable                          | Yes                                         | Seamless Integration                        | Moderate                                        | Fast                                        | Wide Range of Programming Errors and Formatting Errors |
| [flake8](https://github.com/PyCQA/flake8)     | Linter                         | Code Analysis                               | Highly Configurable                          | No                                          | Integration via Plugins                     | Extensive (Adjustable via .flake8 file)         | Moderate                                    | Wide Range of Programming Errors            |
| [pylint](https://github.com/pylint-dev/pylint)| Linter                         | Code Analysis                               | Highly Configurable                          | No                                          | Integration with IDEs, Text Editors         | Extensive (Configuration via .pylintrc)         | Moderate                                    | Wide Range of Programming Errors            |
| [mypy](https://github.com/python/mypy)        | Static Type Checker            | Static Type Checking                        | Configurable                                 | No                                          | Integration with IDEs, Text Editors         | Customizable (Configuration via .mypy.ini)      | Moderate                                    | Type Errors                                 |

### Common Commands

| Feature                                             | __Installation__                            | __pyproject.toml Config__                   | __Other Configs__                            | __Format Code__                                 | __Check Code__                              |
|-----------------------------------------------------|---------------------------------------------|---------------------------------------------|----------------------------------------------|-------------------------------------------------|---------------------------------------------|
| [isort](https://pycqa.github.io/isort/)             | `pip install isort`                         | `[tool.isort]`                              | `setup.cfg, tox.ini, .pep8, .flake8`         | `isort <files_or_dir>`                           | N/A                                         |
| [black](https://black.readthedocs.io/en/stable/)    | `pip install black`                         | `[tool.black]`                              | N/A                                          | `black <files_or_dir>`                           | N/A                                         |
| [yapf](https://github.com/google/yapf)              | `pip install yapf`                          | `[tool.yapf]`                               | `.style.yapf, setup.cfg`                     | `yapf <files_or_dir> -i -r`                      | `yapf <files_or_dir> -r`                     |
| [autopep8](https://github.com/hhatto/autopep8)      | `pip install autopep8`                      | `[tool.autopep8]`                           | `setup.cfg, tox.ini, .pep8, .flake8`         | `autopep8 <files_or_dir> --in-place --recursive` |`autopep8 <files_or_dir> --recursive`         |
| [ruff](https://docs.astral.sh/ruff/)                | `pip install ruff`                          | `[tool.ruff]`                               | `.ruff.toml`                                 | `ruff format <files_or_dir>`                      | `ruff check <files_or_dir>`                        |
| [flake8](https://flake8.pycqa.org/en/latest/)       | `pip install flake8`                        | N/A                                         | `.flake8`                                    | N/A                                             | `flake8 <files_or_dir>`                      |
| [pylint](https://pylint.pycqa.org/en/latest/)       | `pip install pylint`                        | `[tool.yapf]`                               | `.pylintrc`                                  | N/A                                             | `pylint <files_or_dir>`                      |
| [mypy](https://www.mypy-lang.org/)                  | `pip install mypy`                          | `[tool.mypy]`                               | `.mypy.ini`,`mypy.ini`,, `setup.cfg`         | N/A                                             | `mypy <files_or_dir>`                        |

??? tip "More Options for isort"

    - Sorting Imports with isort Using the Black Profile

        To sort import statements using isort with the Black profile, run:

        ```bash
        isort --profile black
        ```

??? tip "More Options for YAPF"

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

!!! note "Other Python Code Formatters and Sorters"

    In addition to Black, isort, and YAPF, there are several other Python code formatters and sorters available, each with its own set of features and capabilities. Some notable mentions include:

    - __pyfmt__: A code formatter that aims to balance flexibility and readability, offering customizable formatting options.
    - __pycodestyle__: A tool that checks Python code against the PEP 8 style guide and provides suggestions for improvements.
    - __blue__: A tool created for the sole reason, of using single quote wherever possible. I prefer double quote, so i dont give in much thought

## Supercharge Your CI with Formatters

!!! example "Combining isort and Black"

    You can combine isort and Black to format and sort your Python files efficiently:

    ```bash
    #!/bin/bash
    isort . --profile black
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

- [Black and Blues - lewoudar.medium.com](https://lewoudar.medium.com/black-and-blues-a5e92e047487)
- [Implement autoformat calpabilities - ruff github issue](https://github.com/astral-sh/ruff/issues/1904)
- [Some notes on the right formater to use for your project - github.com/orsinium](https://github.com/orsinium/notes/blob/master/notes-python/black.md)
- [On testing flake8 - blog - makeuseof.com](https://www.makeuseof.com/python-lint-code-using-flake8/)
- [Seems flake8 is better ?! - reddit](https://www.reddit.com/r/Python/comments/82hgzm/any_advantages_of_flake8_over_pylint/)

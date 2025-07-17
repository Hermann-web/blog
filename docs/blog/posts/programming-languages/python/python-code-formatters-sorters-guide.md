---
date: 2024-02-12
authors: [hermann-web]
comments: true
description: >
  Introduction to Python code formatters and linters including Black, isort, YAPF, ruff, flake8, pylint and more, with usage examples and advanced configurations. Learn how to supercharge your CI with these tools for efficient code formatting and analysis.
categories:
  - programming
  - python
  - code-quality
  - tools-comparison
title: "Exploring Python Code Formatters and Linters: black vs. flake8 vs. isort vs. autopep8 vs. yapf vs. pylint vs. ruff and more"
---

# Exploring Python Code Formatters and Linters: black vs. flake8 vs. isort vs. autopep8 vs. yapf vs. pylint vs. ruff and more

## Introduction

__Are you struggling to maintain consistent formatting in your Python code? Do you find yourself spending too much time organizing imports or adjusting code style manually?__

 <div class="float-img-container float-img-right">
  <a href="https://ibb.co/9th9Q2w"><img src="https://i.ibb.co/FVXbRJs/python-code-carbon.png" alt="python-code-carbon" border="0"></a>
</div>

Navigating the landscape of Python code formatters and linters can be overwhelming, especially for beginners.

This guide serves as your roadmap to mastering Python code formatters and linters, simplifying the process and providing practical examples for effective code formatting, organization, and analysis.

<!-- more -->

Whether you're a novice Python developer or an experienced programmer looking to streamline your workflow, this document is tailored to demystify Python code formatters and linters, offering insights into popular tools like Black, isort, YAPF, ruff, flake8, and pylint. By the end, you'll have a solid understanding of how to leverage these tools to enhance the readability and maintainability of your Python codebase.

## Overview

Python code formatters and linters are tools designed to improve the readability and maintainability of Python code by enforcing consistent formatting and organization standards, while also detecting potential issues and code smells. These tools automatically format your code according to predefined rules and analyze it for potential problems, saving time and ensuring adherence to best practices. In this document, we'll explore popular Python code formatters and linters, including Black, isort, YAPF, ruff, flake8, pylint, and more.

## Key Considerations

### Choosing the Right Tool

- __Code Formatting vs. Code Analysis:__ Distinguish between tools primarily focused on formatting (black, autopep8, yapf, isort) and those focused on analysis (flake8, pylint, ruff).
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

### Common Commands

| Feature                                             | __Installation__                            | __pyproject.toml Config__                   | __Other Configs__                            | __Format Code__                                 | __Check Code__                              |
|-----------------------------------------------------|---------------------------------------------|---------------------------------------------|----------------------------------------------|-------------------------------------------------|---------------------------------------------|
| [isort](https://pycqa.github.io/isort/)             | `pip install isort`                         | `[tool.isort]`                              | `setup.cfg, tox.ini, .pep8, .flake8`         | `isort <files_or_dir>`                           | N/A                                         |
| [black](https://black.readthedocs.io/en/stable/)    | `pip install black`                         | `[tool.black]`                              | N/A                                          | `black <files_or_dir>`                           | N/A                                         |
| [yapf](https://github.com/google/yapf)              | `pip install yapf`                          | `[tool.yapf]`                               | `.style.yapf, setup.cfg`                     | `yapf <files_or_dir> -i -r`                      | `yapf <files_or_dir> -r`                     |
| [autopep8](https://github.com/hhatto/autopep8)      | `pip install autopep8`                      | `[tool.autopep8]`                           | `setup.cfg, tox.ini, .pep8, .flake8`         | `autopep8 <files_or_dir> --in-place --recursive` |`autopep8 <files_or_dir> --recursive`         |
| [ruff](https://docs.astral.sh/ruff/)                | `pip install ruff`                          | `[tool.ruff]`                               | `.ruff.toml`                                 | `ruff format <files_or_dir>`                      | `ruff check <files_or_dir>`                        |
| [flake8](https://flake8.pycqa.org/en/latest/)       | `pip install flake8`                        | N/A                                         | `.flake8`                                    | N/A                                             | `flake8 <files_or_dir>`                      |
| [pylint](https://pylint.pycqa.org/en/latest/)       | `pip install pylint`                        | `[tool.pylint]`                             | `.pylintrc`                                  | N/A                                             | `pylint <files_or_dir>`                      |

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

??? tip "More Options for Ruff"

    - **Specific Rule Selection:** You can enable or disable specific rules:

        ```bash
        ruff check --select E4,E7,E9,F
        ```

    - **Fix Issues Automatically:** Ruff can automatically fix many issues:

        ```bash
        ruff check --fix
        ```

    - **Watch Mode:** Run ruff in watch mode to continuously check files:

        ```bash
        ruff check --watch
        ```

!!! note "Other Python Code Formatters and Linters"

    In addition to the tools mentioned above, there are several other Python formatters and linters available:

    - __pyfmt__: A code formatter that aims to balance flexibility and readability, offering customizable formatting options.
    - __pycodestyle__: A tool that checks Python code against the PEP 8 style guide and provides suggestions for improvements.
    - __blue__: A tool created for the sole reason of using single quotes wherever possible.
    - __bandit__: A security linter that scans Python code for common security issues.
    - __vulture__: Finds unused code in Python programs.

## Supercharge Your CI with Formatters and Linters

!!! example "Combining isort, Black, and Ruff"

    You can combine multiple tools to format and analyze your Python files efficiently:

    ```bash
    #!/bin/bash
    # Format and sort imports
    isort . --profile black
    black . -l 100
    
    # Lint and check for issues
    ruff check .
    flake8 .
    ```

    This script first sorts imports with isort, formats code with Black, and then runs comprehensive linting with ruff and flake8.

??? example "Example with find + Multiple Tools"

    To recursively format and check Python files in a directory and its subdirectories, while excluding certain directories like 'env' and '.git':

    ```bash
    #!/bin/bash
    # Format Python files
    find . -type f -name "*.py" ! -path "*env/*" ! -path "*.git/*" -exec isort {} +
    find . -type f -name "*.py" ! -path "*env/*" ! -path "*.git/*" -exec black -l 100 {} +
    
    # Check for issues
    ruff check .
    flake8 .
    pylint --recursive=y .
    ```

    These commands will find all Python files in the current directory and its subdirectories, excluding specific directories like 'env' and '.git', and then apply formatting and comprehensive linting.

!!! example "GitHub Actions CI Configuration"

    Here's a complete GitHub Actions workflow for code formatting and linting:

    ```yaml
    name: Code Quality

    on: [push, pull_request]

    jobs:
      code-quality:
        runs-on: ubuntu-latest
        steps:
        - uses: actions/checkout@v3
        
        - name: Set up Python
          uses: actions/setup-python@v4
          with:
            python-version: '3.9'
            
        - name: Install dependencies
          run: |
            python -m pip install --upgrade pip
            pip install black isort ruff flake8 pylint
            
        - name: Run formatters
          run: |
            isort --check-only --profile black .
            black --check .
            
        - name: Run linters
          run: |
            ruff check .
            flake8 .
            pylint --recursive=y .
    ```

??? example "pyproject.toml Configuration"

    Comprehensive configuration for multiple tools:

    ```toml
    [tool.black]
    line-length = 100
    target-version = ['py39']
    include = '\.pyi?$'
    extend-exclude = '''
    /(
      # directories
      \.eggs
      | \.git
      | \.hg
      | \.mypy_cache
      | \.tox
      | \.venv
      | build
      | dist
    )/
    '''

    [tool.isort]
    profile = "black"
    line_length = 100
    multi_line_output = 3
    include_trailing_comma = true
    force_grid_wrap = 0
    use_parentheses = true
    ensure_newline_before_comments = true

    [tool.ruff]
    line-length = 100
    target-version = "py39"
    select = [
        "E",  # pycodestyle errors
        "W",  # pycodestyle warnings
        "F",  # pyflakes
        "I",  # isort
        "N",  # pep8-naming
        "UP", # pyupgrade
        "YTT", # flake8-2020
        "S",  # bandit
        "BLE", # flake8-blind-except
        "B",  # flake8-bugbear
        "A",  # flake8-builtins
        "C4", # flake8-comprehensions
        "T10", # flake8-debugger
        "ISC", # flake8-implicit-str-concat
        "ICN", # flake8-import-conventions
        "PT",  # flake8-pytest-style
        "Q",   # flake8-quotes
        "RET", # flake8-return
        "SIM", # flake8-simplify
        "TID", # flake8-tidy-imports
        "ARG", # flake8-unused-arguments
        "ERA", # eradicate
        "PGH", # pygrep-hooks
        "PLC", # pylint conventions
        "PLE", # pylint errors
        "PLR", # pylint refactor
        "PLW", # pylint warnings
        "RUF", # ruff-specific rules
    ]
    ignore = [
        "E501",  # Line too long (handled by black)
        "S101",  # Use of assert
        "PLR0913", # Too many arguments to function call
    ]
    
    [tool.ruff.per-file-ignores]
    "tests/*" = ["S101", "PLR2004"]
    
    [tool.pylint.messages_control]
    disable = [
        "line-too-long",
        "too-many-arguments",
        "too-many-locals",
        "too-few-public-methods",
        "missing-module-docstring",
        "missing-class-docstring",
        "missing-function-docstring",
    ]
    
    [tool.pylint.format]
    max-line-length = 100
    ```

## Best Practices

### Tool Selection Strategy

1. __Start Simple__: Begin with `black` for formatting and `ruff` for linting
2. __Add Gradually__: Introduce additional tools like `isort` for import sorting
3. __Team Consistency__: Ensure all team members use the same configuration
4. __CI Integration__: Run formatters and linters in your CI pipeline

### Performance Optimization

1. __Use Ruff__: Consider `ruff` as a faster alternative to multiple tools
2. __Parallel Execution__: Run different tools in parallel in CI
3. __Incremental Checking__: Use tools that support checking only changed files
4. __Caching__: Enable caching in CI to speed up repeated runs

### Configuration Management

1. __Centralized Config__: Use `pyproject.toml` for tool configuration
2. __Consistent Settings__: Align settings across tools (e.g., line length)
3. __Version Control__: Include configuration files in version control
4. __Documentation__: Document tool choices and configurations for your team

## Conclusion

Python code formatters and linters are essential tools for maintaining high-quality, consistent codebases. By combining formatters like Black and isort with linters like ruff, flake8, and pylint, developers can ensure their code adheres to established standards and best practices, leading to more efficient and collaborative development processes.

The key is to start with essential tools and gradually build up your toolchain based on your project's needs. Modern tools like ruff are making it easier to get comprehensive code analysis with minimal configuration, while established tools like Black continue to provide reliable, opinionated formatting.

## Related Posts

- Read [Python Type Checking Tools: mypy vs. pyright vs. pydantic vs. pandera vs. jaxtyping vs. check_shapes vs. typeguard](./python-type-checking-tools.md)
- [Cheat on Python Package Managers](./package-managers-in-python.md)

## Relevant Sources

- [Black and Blues - lewoudar.medium.com](https://lewoudar.medium.com/black-and-blues-a5e92e047487)
- [Implement autoformat capabilities - ruff github issue](https://github.com/astral-sh/ruff/issues/1904)
- [Some notes on the right formatter to use for your project - github.com/orsinium](https://github.com/orsinium/notes/blob/master/notes-python/black.md)
- [On testing flake8 - blog - makeuseof.com](https://www.makeuseof.com/python-lint-code-using-flake8/)
- [Seems flake8 is better ?! - reddit](https://www.reddit.com/r/Python/comments/82hgzm/any_advantages_of_flake8_over_pylint/)
- [Ruff Documentation - docs.astral.sh](https://docs.astral.sh/ruff/)
- [Black Documentation - black.readthedocs.io](https://black.readthedocs.io/en/stable/)
- [isort Documentation - pycqa.github.io/isort](https://pycqa.github.io/isort/)
- [Flake8 Documentation - flake8.pycqa.org](https://flake8.pycqa.org/en/latest/)
- [Pylint Documentation - pylint.pycqa.org](https://pylint.pycqa.org/en/latest/)

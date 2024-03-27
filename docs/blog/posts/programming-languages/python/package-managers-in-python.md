---
date: 2023-10-28
authors: [hermann-web]
comments: true
description: >
  Delve into the intricacies of managing Python dependencies efficiently using pip, pipenv, poetry, conda and more.
categories:
  - programming
  - python
  - package-manager
  - tools-comparison
  - project-management
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practices/software-licenses.md
title: "Managing Python Dependencies: Navigating pip, pipenv, poetry, conda and more"
---

# Managing Python Dependencies: Navigating pip, pipenv, poetry, conda and more

## Introduction

In the realm of Python development, a crucial aspect is managing project dependencies effectively.

<div class="float-img-container float-img-right">
  <a title="www.python.org, GPL &lt;http://www.gnu.org/licenses/gpl.html&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Python-logo-notext.svg"><img width="64" alt="Python-logo-notext" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/64px-Python-logo-notext.svg.png"></a>
</div>

This guide delves into four prominent tools—pip, pipenv, poetry, conda and more—each offering distinct approaches to dependency management.
Grasping their strengths, weaknesses, and use cases empowers you to make informed decisions for your projects.

<!-- more -->

## Key Considerations

### Choosing the Right Tool

- **Scope and Size:** Consider project scale and complexity.
- **Team Collaboration:** Assess the extent of teamwork involved.
- **Environment Isolation:** Determine the need for isolated environments.
- **Dependency Management:** Evaluate desired features like dependency locking and version conflict resolution.
- **Project Requirements:** Factor in specific project needs or constraints.

### Tools Overview

=== ":octicons-file-code-16: `pip`"

    - **Core Package Installer:** The foundation for Python package management.
    - **Global Installations:** Installs packages system-wide by default.
    - **Virtual Environments:** Can be used within virtual environments for isolation.
    - **Basic Dependency Management:** Relies on `requirements.txt` for specifying dependencies.

=== ":octicons-file-code-16: `pipenv`"

    - **Virtual Environment Creation and Management:** Automatically creates and manages virtual environments.
    - **Dependency Locking:** Locks dependency versions for reproducibility.
    - **Integration with Pipfile:** Uses Pipfile and Pipfile.lock for dependency management.

=== ":octicons-file-code-16: `poetry`"

    - **Dependency Management and Packaging:** Comprehensive dependency management and packaging tool.
    - **Virtual Environment Handling:** Manages virtual environments effectively.
    - **Dependency Locking and Version Handling:** Offers robust dependency locking and version conflict resolution.
    - **Declarative Syntax:** Uses pyproject.toml for configuration.

=== ":octicons-file-code-16: `conda`"

    - **Cross-Platform Package and Environment Management:** Manages packages and environments across multiple languages (Python, R, etc.).
    - **Large Ecosystem of Packages:** Accesses a vast repository of packages through Anaconda repositories.
    - **Environment Isolation:** Creates isolated environments for project-specific dependencies.
    - **Non-Python Dependencies:** Handles non-Python dependencies as well.

=== ":octicons-file-code-16: `uv`"

    - **Extremely Fast Installation:** Faster than traditional pip and pip-tools.
    - **Drop-in Replacement:** Provides a familiar interface for common pip commands.
    - **Supports Advanced Features:** Handles editable installs, Git dependencies, URL dependencies, etc.
    - **Dependency Management:** Offers features like dependency overrides and conflict resolution.
    - **Limitations:** Not a complete replacement for pip; some features are still missing.
    - **Platform-specific Requirements File Generation:** Generates requirements files tailored to specific platforms.

## Comparison Table

| Feature                        | [pip](https://pypi.org/project/pip/)                                       | [pipenv](https://github.com/pypa/pipenv)                                    | [poetry](https://github.com/python-poetry/poetry)                                      | [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)                                       | [uv](https://github.com/astral-sh/uv)                                         |
|--------------------------------|-------------------------------------------|--------------------------------------------|---------------------------------------------|---------------------------------------------|--------------------------------------------|
| Installation                   | Built-in                                  | `pip install pipenv`                       | `pip install poetry`                        | Download and install Anaconda or Miniconda  | `pip install uv`                     |
| Environments                   | Implicit                                  | Automatic                                  | Automatic                                   | Automatic                                   | Implicit                                  |
| Locking                        | Manual                                    | Automatic                                  | Automatic                                   | Automatic                                   | Manual                                  |
| Configuration                  | requirements.txt                          | Pipfile, Pipfile.lock                       | pyproject.toml                              | Environment files                           | N/A                                        |
| Official packages              | [PyPI (Python Package Index)](https://pypi.org/)                | [PyPI](https://pypi.org/)                                       | [PyPI](https://pypi.org/)                                        | [Conda Forge (conda-forge channel)](https://anaconda.org/conda-forge)           | [PyPI (Python Package Index)](https://pypi.org/)                 |
| Multi-python-version Resolution   | No                                        | No                                         | Yes                                         | No                                          | Yes                                        |

### Common Commands

| Feature               | pip                                       | pipenv                                    | [poetry](https://python-poetry.org/docs/cli)                                      | conda                                       | uv                                         |
|-----------------------|-------------------------------------------|--------------------------------------------|---------------------------------------------|---------------------------------------------|--------------------------------------------|
| **Create a project**  | `python -m venv <env-name>`               | `pipenv --python <python-version>`          | `poetry init`                               | `conda create -n <env-name> python=<python-version>` | `uv venv` (creates virtualenv at: `.venv/`)                                  |
| **Use a virtual environment** | `source <env-name>/bin/activate` (linux) or `<env-name>/Scripts/activate` (windows) | `pipenv shell`                       | `poetry shell`                              | `conda activate <env-name>`                   | Same as pip                                        |
| **Install a package** | `pip install <package-name>`              | `pipenv install <package-name>`             | `poetry add <package-name>`                 | `conda install <package-name>`               | `uv pip install <package-name>`            |
| **Remove a package**  | `pip uninstall <package-name>`            | `pipenv uninstall <package-name>`           | `poetry remove <package-name>`              | `conda uninstall <package-name>`             | `uv pip uninstall <package-name>`          |
| **Install from requirements** | `pip install -r requirements.txt`      | `pipenv install` (reads from Pipfile)       | `poetry install` (reads from pyproject.toml) | `conda install --file requirements.txt`       | `uv pip sync requirements.txt` or `uv pip install -r requirements.txt`           |
| **Dev packages vs others** | No explicit distinction                | `--dev` flag for development dependencies  | `[tool.poetry.dev-dependencies]` section in pyproject.toml | No explicit distinction | Not currently supported (planned for future) |
| **List requirements**  | `pip freeze > requirements.txt` or use `pipreqs`            | `pipenv lock -r > requirements.txt`        | `poetry export -f requirements.txt > requirements.txt` | `conda list --export > requirements.txt`     | Same as pip (e.g., `uv pip freeze > requirements.txt`) |
| **Transport project**  | Manually copy files or create a setup.py   | Copy Pipfile and Pipfile.lock               | Copy pyproject.toml and pyproject.toml.lock  | Export environment to .yml file (conda env export > environment.yml) | Not directly supported (requires rebuilding with UV) |
| **Install from lock file** | N/A                                   | `pipenv install --ignore-pipfile`           | `poetry install --no-dev` (reads from pyproject.toml.lock) | `conda env update --file environment.yml`       | N/A (Dependency resolution handled by UV directly) |
| **Delete the venv**   | Remove directory manually                  | `pipenv --rm`                              | `poetry env remove <env-name>`              | `conda remove --name <env-name> --all`      | N/A                                        |
| **Update a package to its latest version** | `pip install <package-name> --upgrade`   | `pipenv update <package-name>`              | `poetry update <package-name>`              | `conda update <package-name>`               | `uv pip install <package-name> --upgrade` |
| **Update all packages to their latest versions** | N/A                                 | `pipenv update`                             | `poetry update`                             | `conda update -n <env-name> --all` (be aware of potential conflicts)                  | N/A                                        |

<!-- ### Additional Features

| Feature               | pip                                       | pipenv                                    | poetry                                      | conda                                       | uv                                         |
|-----------------------|-------------------------------------------|--------------------------------------------|---------------------------------------------|---------------------------------------------|--------------------------------------------|
| **Python Discovery**  |                                           |                                            |                                             |                                             |                                              |
| Virtual Environment   | `python -m venv <env-name>`, then `source <env-name>/bin/activate` | `pipenv shell`                       | `poetry shell`                              | `conda activate <env-name>`                   | N/A                                        |
| Python Interpreter    | System-wide                                | System-wide                                | System-wide                                 | System-wide                                 | VIRTUAL_ENV, CONDA_PREFIX, .venv, python3/python.exe |
| **Git Authentication**|                                           |                                            |                                             |                                             |                                              |
| Authentication Methods | Username/Password, SSH Keys, Tokens      | Username/Password, SSH Keys, Tokens        | Username/Password, SSH Keys, Tokens         | Username/Password, SSH Keys, Tokens         | Username/Password, SSH Keys, Tokens        |
| **Dependency Caching**|                                           |                                            |                                             |                                             |                                              |
| Caching Semantics     | HTTP Headers, URL, Commit Hash            | HTTP Headers, URL, Commit Hash            | HTTP Headers, URL, Commit Hash              | HTTP Headers, URL                          | HTTP Headers, URL, Commit Hash             |
| **Resolution Strategy**|                                           |                                            |                                             |                                             |                                              |
| Default Strategy      | Latest Compatible Version                 | Latest Compatible Version                 | Latest Compatible Version                   | Latest Compatible Version                   | Latest Compatible Version                   |
| Alternative Strategy  | N/A                                       | N/A                                        | N/A                                         | N/A                                         | Lowest Compatible Version, Lowest Compatible Direct | -->

!!! note "Remember !"
    As you wrap up your exploration of Python's dependency management tools, remember:

    - **Consistency:** Maintain a uniform approach within a project for easier maintenance.
    - **Descriptive Naming:** Use clear and informative names for environments and dependencies.
    - **Thorough Testing:** Ensure compatibility and functionality after dependency changes.
    - **Regular Updates:** Keep tools and dependencies up-to-date for security and performance.

??? info "More on UV dependencies resolution"

    - **Generate Requirements from pyproject.toml:**
        ```bash
        uv pip-compile pyproject.toml -o requirements.txt
        ```
    - **Platform-specific Requirements Generation:**
        ```bash
        uv pip-compile requirements.in -o requirements.txt
        ```

    - **Command Line Compatibility:**
      uv's `pip-install` (`uv pip`) and `pip-compile` (`uv compile`) commands support many familiar command-line arguments, such as `-r requirements.txt`, `-c constraints.txt`, `-e .` (for editable installs), `--index-url`, and more.

    - **Resolution Strategy Customization:**
      uv allows customization of its resolution strategy. With `--resolution=lowest`, uv installs the lowest compatible versions for all dependencies, while `--resolution=lowest-direct` focuses on lowest compatible versions for direct dependencies and latest compatible versions for transitive dependencies.

??? info "read [this docu](./integrating-requirements-with-poetry.md) about how to add dependencies from a `requirements.txt` file"

?? question "How to use two conda environment at the same time without breaking anything ?"

    To simultaneously utilize two Conda environments without causing conflicts, you can leverage Conda's nested activation feature. Here's how you can achieve this:

    ```bash
    conda activate old_project_env
    conda activate --stack new_project_env
    ```

    This sequence of commands allows you to activate a parent environment (`old_project_env`) while retaining access to the command-line options from the old environment. Then, by using `conda activate --stack new_project_env`, you can activate a new environment (`new_project_env`) within the context of the parent environment.

    This approach ensures that you can work within the specified environment (`new_project_env`) without interfering with the configurations or dependencies of the parent environment (`old_project_env`).

    For more information on Conda environments and nested activation, you can refer to the [official Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#nested-activation).
    ```

## Conclusion

Congratulations! You've now equipped yourself with insights into managing Python dependencies effectively using pip, pipenv, poetry, conda and more. By understanding their strengths and use cases, you're better equipped to make informed decisions for your projects.

Stay proactive in exploring and adapting these tools to optimize your Python development experience!

## Related Links

- [is it safe to update all conda package at once ?](https://stackoverflow.com/a/44072944)

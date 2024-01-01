---
date: 2023-12-17
authors: [hermann-web]
description: >
  Discover MkDocs, the tool that makes documentation creation straightforward. Explore its simplicity in crafting professional documentation for your projects.
categories:
  - devops
  - mkdocs
  - beginners
  - documentation
title: "Getting Started with MkDocs for Documentation"
subtitle: "MkDocs: Your Straightforward Documentation Companion"
---


# MkDocs: Your Straightforward Documentation Companion

## Introduction

__Welcome to MkDocs: the hassle-free documentation solution!__

In search of a tool that makes documentation creation a breeze? [MkDocs](https://github.com/mkdocs/mkdocs/) is your answer!

<!-- ![](./assets/mkdocs-logo.png) -->
<div class="float-img-container float-img-left">
  <a title="mkdocs-material" href="https://github.com/squidfunk/mkdocs-material">
    <img style="width: 9rem;" alt="mkdocs-logo" src="https://raw.githubusercontent.com/squidfunk/mkdocs-material/master/.github/assets/logo.svg">
  </a>
</div>

This straightforward platform simplifies the process of generating professional project documentation.

This guide is your gateway to exploring MkDocs' user-friendly approach. You'll uncover how this tool streamlines the creation of polished and organized documentation for all your projects. Let's dive in and harness MkDocs' straightforwardness for your documentation needs.


<!-- more -->

## Comparing Documentation Tools

When it comes to documenting projects, various tools offer unique features and complexities. Let's explore a few:

- **Sphinx**: Known for its robustness and flexibility, Sphinx is powerful but can be intricate for beginners due to its configuration requirements.

- **Docusaurus**: Ideal for creating user-centric documentation with React, but might feel overwhelming for those unfamiliar with JavaScript frameworks.

- **GitBook**: Offers a user-friendly interface, yet its extensive feature set might be more than needed for straightforward documentation needs.

- **MkDocs**: Unlike some other tools, MkDocs stands out for its simplicity. It's based on Markdown, a plain text format, making it incredibly easy to use. With MkDocs, creating professional documentation feels straightforward and hassle-free.

## MkDocs: Embracing Simplicity

MkDocs adopts Markdown, a plain text format widely accessible and intuitive for beginners. Its minimalistic approach enables users to focus on content creation without getting lost in complex configurations.

### Features of MkDocs:

- **Simple Configuration**: MkDocs requires minimal setup, with a straightforward configuration file.
- **User-Friendly**: Its Markdown-based structure simplifies content creation for all levels of users.
- **Live Preview**: Offers a live preview of documentation, ensuring instant visual feedback.
- **Extensibility**: While basic, MkDocs supports various themes and plugins for enhanced functionality.

MkDocs excels in providing a straightforward and efficient way to create professional documentation without overwhelming users with unnecessary complexities. It's the perfect choice for those seeking a quick and easy documentation solution.

## Tutorial: Getting Started with MkDocs

MkDocs simplifies the process of creating documentation for your Python projects. Follow these steps to create a documentation site using MkDocs:

### 1. **Install MkDocs**:

Install MkDocs by running the following command in your terminal:

```bash
pip install mkdocs
```

### 2. **Set Up Your Project**:

Create a new directory for your project and initialize an MkDocs project:

```bash
mkdir my-project
cd my-project
mkdocs new .
```

This creates a new `mkdocs.yml` configuration file and a `docs` directory with a sample Markdown file.

### 3. **Install a Theme**:

Enhance your documentation's appearance by installing a theme like [`mkdocs-material`](https://github.com/squidfunk/mkdocs-material):

```bash
pip install mkdocs-material
```

### 4. **Configure Your Site**:

Edit the `mkdocs.yml` file to configure your documentation site. Define the title, theme, and pages to include. Check [examples](https://github.com/boisgera/pandoc/blob/master/mkdocs.yml).

- Configure `docs_dir` to specify the folder where MkDocs will find `.md` files.
- Use the `nav` section to structure your files into tabs.

### 5. **Write Documentation**:

Create your documentation in Markdown format and save the files in the `docs` directory.

### 6. **Preview Your Site**:

To preview your site locally, run:

```bash
mkdocs serve
```
This will start a local web server and open your documentation site in your default web browser. You can make changes to your documentation and the site will automatically update.

### 7. **(optional) More options**:

You can add more options. For example,
```bash
mkdocs serve --dirty -a localhost:8001
```

!!! note "Note"
    - `--dirty`: Only re-build files that have changed.
    - `-a, --dev-addr <IP:PORT>`: IP address and port to serve documentation locally (default: localhost:8000)
    - use `mkdocs serve -h` for more options

!!! warning "warning"
    `A 'dirty' build [...] will likely lead to inaccurate navigation and other links within your site. This option is designed for site development purposes only.`, mkdocs

### 8. **Build Your Site**:

Generate a static HTML site by running:

```bash
mkdocs build
```

This creates a `site` directory containing the built site. You can deploy this to a web server.

Remember, MkDocs supports numerous plugins, such as `mkdocs-run-shell-cmd-plugin`, enabling extended functionalities.

## Conclusion

MkDocs provides a straightforward way to create and manage documentation for Python projects. With its simple setup, configuration, and live preview features, it streamlines the documentation process, making it an excellent choice for developers.

## More ressources

Explore the resources below to dive deeper into MkDocs:

- [Real Python - Python Project Documentation with MkDocs](https://realpython.com/python-project-documentation-with-mkdocs/)
- [MkDocs - Getting Started](https://www.mkdocs.org/getting-started/)
- [MkDocs - User Guide CLI](https://www.mkdocs.org/user-guide/cli/)
- [MkDocs - Issues and Discussions](https://github.com/mkdocs/mkdocs/issues/2186)
- [YouTube - Getting Started with MkDocs](https://youtube.com/watch?v=Q9wMAv5airg)
- [MkDocs Run Shell Cmd Plugin](https://pypi.org/project/mkdocs-run-shell-cmd-plugin/)

## Related Pages

- [MkDocs: Your Straightforward Documentation Companion](./mkdocs-get-started.md)

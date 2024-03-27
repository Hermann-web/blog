---
date: 2024-01-04
authors: [hermann-web]
comments: true
description: |
  Simplifying Large File Management in Git with Git LFS: A Comprehensive Guide for Developers.
categories:
  - version-control
  - git
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: "Simplifying Large File Management in Git with Git LFS"
---

# Simplifying Large File Management in Git with Git LFS

## Introduction

## Introduction

__Have you ever faced the challenge of managing large files within a Git repository?__

<div class="float-img-container float-img-right">
  <a title="Credit: git-lfs.com" href="https://git-lfs.com/"><img alt="git-lfs-logo" src="https://git-lfs.com/images/graphic.gif"></a>
</div>

Whether you're an experienced developer or just beginning your coding journey, dealing with large files in version control can be perplexing. Often, developers resort to `.gitignore` to exclude files, but what if there are essential large files crucial for your project's integrity?

Enter [Git LFS](https://git-lfs.com) (Large File Storage), a solution designed to revolutionize how Git repositories handle large files. While some files are pivotal to track, keeping repositories lean and efficient remains a priority.

<!-- more -->

This guide unlocks the potential of Git LFS, providing a step-by-step approach to seamlessly incorporate it into your version control workflow. Discover how Git LFS streamlines large file management, ensuring your repository stays clean and optimized.

Let's navigate the realm of large file management in Git, ensuring your projects stay organized and efficient.

## Steps to Implement Git LFS

### 1. Installing Git LFS

Begin by installing Git LFS. Visit the [Git LFS website](https://github.com/git-lfs/git-lfs/wiki/Installation) for installation instructions tailored to your operating system.

### 2. Initializing Git LFS

In your repository, run the command:

```bash
git lfs install
```

This command sets up Git LFS for your project, preparing it to manage large files.

### 3. Tracking Large Files

Identify the large files you want to store using Git LFS and begin tracking them. You can manually specify these files using:

```bash
git lfs track "path/to/your/large/file"
```

??? example "Example: Track files `.avi` and `.gif` files larger than 19MB"

        For an efficient approach to track multiple files of specific extensions and sizes (such as `.avi` and `.gif` files larger than 19MB), a script can simplify the process. For exa:

        ```bash
        #!/bin/bash

        # Find .avi files larger than 19MB and track them with Git LFS
        find . -type f -name "*.avi" -size +19M | while read -r file; do
            git lfs track "$file"
        done

        # Find .gif files larger than 19MB and track them with Git LFS
        find . -type f -name "*.gif" -size +19M | while read -r file; do
            git lfs track "$file"
        done
        ```

        Ensure to execute this script within your Git repository directory. 

!!! Warning "Always review in `.gitattributes` the file selections to confirm they match your requirements before committing changes to Git LFS."

!!! note "You should also note there is a quota limit. read more [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-storage-and-bandwidth-usage)"

### 4. Updating `.gitattributes`

After tracking the files, update your `.gitattributes` file with the tracking information:

```bash
git add .gitattributes
git commit -m "Track large .avi and .gif files with Git LFS"
```

### 5. Commit and Push Changes

Following the usual Git workflow, add and commit your changes:

```bash
git add .
git commit -m "Message"
```

Finally, push the changes to your remote repository:

```bash
git push origin master
```

Assuming you're on the master branch, this step uploads the large files to the Git LFS server.

By successfully implementing Git LFS, your large files are now efficiently managed within the repository, enhancing version control capabilities.

For detailed installation instructions and additional information about Git LFS, refer to the [Git LFS installation guide](https://github.com/git-lfs/git-lfs/wiki/Installation).

---
date: 2023-11-14
authors: [hermann-web]
comments: true
description: |
  Dive into the world of Git submodules—a tool that helps manage code pieces within your projects. This guide simplifies the complexities, making submodule management accessible for developers new to the concept.
categories:
  - version-control
  - git
  - beginners
title: "Nesting Repositories with Git Submodules: A Newbie's Guide"
---

## Introduction

__Are you facing the challenge of handling multiple code pieces scattered across different repositories in your project, unsure how to seamlessly integrate them?__

For developers new to the concept, managing disparate repositories within a single project can be overwhelming. Git submodules offer a guiding light, acting as a map through the maze of organizing and linking these separate codebases or libraries within your projects.

### Real-Life Scenario: Aligning Frontend and Backend Strategies

Back in 2022, I found myself as the lead developer overseeing the backend team, while collaborating closely with a talented frontend developer responsible for crafting engaging user interfaces.

Our teams operated independently, each excelling in our specialized domains. However, this independence led to distinct branch strategies. The backend team adopted a unique approach, separate from the frontend's strategy.

Over time, this divergence in branch strategies caused disparities between our repositories' states. Aligning frontend changes with the evolving backend structures became a complex task. Ensuring seamless integration between our frontend branches and specific backend versions posed challenges.

<!-- more -->

Recognizing these challenges, I engaged in a discussion with the frontend developer. We brainstormed solutions to synchronize versions and segregate our branch strategies effectively.

During our deliberation, we explored the idea of utilizing Git submodules. It wasn't merely about syncing versions but aligning our separate branch strategies while maintaining distinct team autonomy.

The proposal envisioned Git submodules as the bridge between our frontend and backend repositories, facilitating version synchronization and accommodating separate yet aligned branch strategies. This approach aimed to streamline collaboration and ensure smoother integration between our teams' work.

Motivated by the vision of enhanced collaboration and harmonized branch strategies, we collectively agreed to integrate Git submodules. This decision promised a more cohesive development environment, allowing both teams to synchronize versions and align branch strategies seamlessly.

### Extending to a Computer Vision Project

Additionally, in a computer vision project, I encountered a similar challenge. Testing code from various repositories required frequent modifications, causing inefficiencies. Managing these disparate codebases led me to prefer a unified repository managed with Git submodules. This approach enabled me to centralize and manage all required codebases efficiently, adapting them as needed.

---

Think of Git submodules as containers that neatly organize and link external repositories to your main project—providing a solution to the discomfort of handling disjointed pieces of code. Join us as we embark on a journey to explore how to clone, set up, and effortlessly synchronize these submodules within your projects.

<a title="Image Source: Wikimedia Commons" href="<your_image_source_link>"><img width="512" alt="Git Submodules" src="<your_image_path>"></a>

This document simplifies Git submodules in a beginner-friendly way, offering developers new to the concept a clear path to effectively manage multiple repositories as cohesive parts of their projects.

<!-- This document serves as a comprehensive resource, elucidating Git submodules, and providing step-by-step guidance to harness their potential in your development endeavors. -->

<!-- Welcome to our comprehensive guide on Git submodules! This guide will take you through the process of working with Git repositories that contain submodules, allowing you to manage dependencies and streamline your development workflow effectively. -->

## Cloning a Repository with Submodules and Cloning a Specific Submodule

### Clone the Main Repository

1. **Open your terminal and navigate to the desired directory for cloning:**

    ```bash
    cd /desired/directory/path
    ```

2. **Clone the main repository:**

    ```bash
    git clone <repository_url>
    ```

    Replace `<repository_url>` with the URL of the main repository.

3. **Change your working directory to the repository:**

    ```bash
    cd <repository_directory>
    ```

    Replace `<repository_directory>` with the name of the cloned directory.

### Initialize and Update Submodules

4. **Initialize the submodules:**

    ```bash
    git submodule init
    ```

    This sets up necessary Git configurations for submodules.

5. **Update the submodules:**

    ```bash
    git submodule update
    ```

    This fetches submodule contents based on references in the main repository.

### Clone a Specific Submodule

6. **Clone a specific submodule:**

    ```bash
    git submodule update --recursive -- <submodule_path>
    ```

    Replace `<submodule_path>` with the specific submodule path. This command updates only the specified submodule and its dependencies, leaving others unchanged.
    - The `--recursive` flag initializes nested submodules within the specified submodule.

Now that you've successfully cloned the main repository along with its submodules, let's explore how to create and manage submodules within an existing repository.

## Creating a Git Submodule

### Navigate to the Parent Repository

1. **Move to the parent repository's root directory:**

    ```bash
    cd /path/to/parent/repository
    ```

2. **Add the Submodule:**

    ```bash
    git submodule add <submodule_repository_url> <submodule_path>
    ```

    - `<submodule_repository_url>`: URL of the submodule repository.
    - `<submodule_path>`: Path within the parent repository to place the submodule.

    Example:

    ```bash
    git submodule add https://github.com/example/submodule-repo.git path/to/submodule
    ```

3. **Commit the Changes:**

    ```bash
    git commit -m "Add submodule: <submodule_path>"
    ```

    Replace `<submodule_path>` with the actual path used when adding the submodule.

4. **Push Changes (Optional):**

    ```bash
    git push
    ```

Now, let's delve into pulling changes from both the main repository and its submodules to keep your local copy up to date.

## Pulling Changes from the Main Repository and Submodules

### Update the Main Repository

1. **Navigate to the main repository's directory:**

    ```bash
    cd /path/to/main/repository
    ```

2. **Fetch the latest changes:**

    ```bash
    git pull origin main
    ```

    This command fetches and merges the latest changes from the remote repository into your local `main` branch.

### Update Submodules

1. **Update submodules to the latest commits:**

    ```bash
    git submodule update --remote
    ```

    This updates each submodule to the commit specified by the main repository.

2. **Update a specific submodule:**
    - Using `git submodule update --remote <submodule_path>`:

        ```bash
        git submodule update --remote path/to/submodule
        ```

    - Or manually in the submodule directory:

        ```bash
        cd path/to/submodule
        git pull origin master
        ```

### Pushing Updated Submodule References (Bonus)

1. **Inside the main repository, after updating submodule references:**

    ```bash
    git commit -am "Update submodule references"
    git push origin main
    ```

2. **If there are changes in the submodules themselves:**

    ```bash
    cd path/to/submodule
    git commit -am "Update submodule"
    git push origin master
    ```

By following these steps, you'll effortlessly manage main repositories and their submodules, ensuring your projects are up to date.

Stay tuned for more Git tips and tricks on our blog for seamless collaboration and version control!

## Related pages

- [Managing Local Modifications and Remote Changes in Git](./pull-changes-with-conflicts.md)
- [Mastering Git Merge Strategies: A Developer's Guide](./sync-branches-with-conflicts.md)
- [Understanding Git Pull vs Merge in Git Workflow](./git-pull-vs-git-merge-equivalence.md)
- [Mastering Git Branch Handling: Strategies for Deletion and Recovery](./handling-branch-deletion.md)

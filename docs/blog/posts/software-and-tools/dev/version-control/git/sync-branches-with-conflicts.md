---
date: 2023-10-11
authors: [hermann-web]
comments: true
description: |
  Discover the best practices in Git merge strategies—rebase and merge. Navigate the maze of version control to maintain a pristine repository history, perfect for developers entering collaborative coding environments.
categories:
  - version-control
  - git
title: "Mastering Git Merge Strategies: A Developer's Guide"
---


## Introduction

__Have you ever found yourself tangled in a web of Git branches, unsure of the best path to weave your changes together?__

The world of version control can be a maze, especially when deciding between Git's merge strategies. Fear not! This guide is your compass through the wilderness of rebases and merges, shedding light on the best routes to keep your repository history tidy and your sanity intact.

<div class="float-img-container float-img-right">
  <a title="Renatasds, CC BY-SA 4.0 &lt;https://creativecommons.org/licenses/by-sa/4.0&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Mergevsrebase.png"><img width="512" alt="Mergevsrebase" src="https://upload.wikimedia.org/wikipedia/commons/f/fe/Mergevsrebase.png"></a>
</div>

Git offers two primary trails: the __rebase__, known for its clean and linear history, and the __merge__, preserving the unique storylines of each branch. Join us on this journey as we navigate the pros, cons, and conflict resolution techniques, empowering you to choose the right path for your project's narrative.

So, this document provides guidance on using Git merge strategies, specifically focusing on the rebase and merge options.

<!-- more -->

### Git Branch Operations Cheat Sheet

Before diving into merging and rebasing, it’s helpful to remember some key branch operations:

| Task (Branch → Target)     | Description                                    | Command / Notes                                    |
| -------------------------- | ---------------------------------------------- | -------------------------------------------------- |
| Create new branch          | Start a new line of development                | `git branch <branch>` / `git switch -c <branch>`   |
| Switch branch              | Move between branches                          | `git switch <branch>`                              |
| Merge branch into current  | Combine another branch’s commits into current  | `git merge <branch>` (fast-forward or 3-way merge) |
| Rebase current onto branch | Replay commits on top of another branch        | `git rebase <branch>`                              |
| Reset branch to commit     | Move branch HEAD to a specific commit          | `git reset --soft/--mixed/--hard <commit>`         |
| Restore branch files       | Restore specific files from a commit or branch | `git restore --source=<branch-or-commit> <file>`   |

---

### Notes on Branch Management

* Merging and rebasing are your primary tools for integrating work from other branches.

  * **Merge** = preserves history, creates a new merge commit
  * **Rebase** = linearizes history, replaying commits on top of target branch

* **`git reset`** and **`git restore`** are still useful in branch workflows:

  * Resetting can move your branch to a specific commit safely in local dev.
  * Restoring files allows you to pull a clean version from another branch without affecting commits.

* When referencing commits in reset or restore, you can use commit hashes or branch names interchangeably.

* Always double-check which branch you’re on—operations like `reset --hard` or `rebase` can irreversibly alter history if applied on the wrong branch.

For a deeper dive into managing files and commits across states, see [**Git File and Commit Operations: Managing Changes Between States**](./commit-operations.md) – _Learn how to navigate files and commits across different Git states for effective version control._

### Rebase: Creating a Linear History

```bash
git checkout feature-branch
git pull --rebase origin main
```

or

```bash
git fetch origin main 
git checkout feature-branch
git rebase origin main 
```

=== ":octicons-file-code-16: `Pros`"

    - Keeps a linear, clean commit history.
    - Integrates local changes after remote ones, maintaining chronological order.

=== ":octicons-file-code-16: `Cons`"

    - Requires manual resolution of conflicts that may arise during rebase.

### Handling Conflicts during Rebase

During the process of rebasing branches, conflicts might arise when applying commits from one branch onto another. Git requires manual resolution of conflicts that occur during a rebase operation.

#### Resolving Conflicts Manually

When conflicts occur during a rebase, Git halts the process and prompts you to resolve conflicts in the files where they arise. After resolving conflicts, you can continue the rebase using:

```bash
git rebase --continue
```

#### VSCode for Conflict Handling

Visual Studio Code (VSCode) offers a user-friendly interface to resolve conflicts during a rebase operation. Follow these steps within the VSCode environment:

1. __Start the Rebase:__ Execute the rebase command in your terminal:

    ```bash
    git rebase <branch_name>
    ```

    This command initiates the rebase process.

2. __Conflict Indication:__ When conflicts occur, VSCode visually highlights them within the editor. You'll notice markers indicating the conflicted sections.

3. __Resolve Conflicts:__ Navigate to the conflicted file(s) in VSCode. Locate the sections marked as conflicted, displaying both versions of the conflicting changes.

4. __Choose Resolution:__ Review the changes and decide which version to keep or edit the content to create a resolution. Remove conflict markers (<<<<<<<, =======, >>>>>>>) once the conflict is resolved.

5. __Stage Changes:__ After resolving conflicts in each file, stage the changes using the Source Control panel in VSCode.

6. __Continue Rebase:__ Once conflicts are resolved and staged, return to your terminal and continue the rebase:

    ```bash
    git rebase --continue
    ```

    This command proceeds with the rebase process using the resolved changes.

VSCode streamlines the conflict resolution process by providing a visual and intuitive interface, making it easier to handle conflicts during a rebase operation.

## Merge: Preserving Branch Narratives

Using `merge` in Git combines changes from different branches, preserving their individual commit histories. This method creates a new commit to capture the integration of changes from one branch into another.

=== ":octicons-file-code-16: `Pros`"

    - Preserves the complete history of changes made in each branch.
    - Maintains a clear track record of when and where changes were integrated.

=== ":octicons-file-code-16: `Cons`"

    - May result in a non-linear history with multiple merge commit points.
    - Can potentially clutter the commit history with merge commits.

### Handling Conflicts

Similar to the rebase operation, merging branches in Git can lead to conflicts, especially when changes made in the same file or code sections conflict with each other. Git provides options to manage these conflicts during a merge operation.

#### Resolving Conflicts by Favoring a Specific Branch

Suppose you're merging `branchA` into `branchB` and wish to favor the changes from `branchB` in case of conflicts:

  ```bash
  git checkout branchB  # Switch to the target branch (branchB)
  git merge -X ours branchA  # Merge branchA into branchB, favoring branchB changes in conflicts
  ```

!!! note "Explanation:"

    1. `git checkout branchB`: Switches to the target branch where changes will be merged (in this case, `branchB`).

    2. `git merge -X ours branchA`: Merges `branchA` into `branchB`, and the `-X ours` option ensures conflicts are resolved by favoring changes from the current branch (`branchB`).

Upon executing this command, Git will merge the changes from `branchA` into `branchB`, automatically resolving conflicts by favoring the modifications present in `branchB`.

### Other Merge Strategies

Git provides various merge strategies such as `recursive`, `octopus`, and `resolve`, each with its own approach to handling merges. Choosing the right strategy depends on the project's requirements and the nature of changes between branches.

## Conclusion

As we conclude this journey through Git's merge strategies, remember the beauty lies in choice. Rebase crafts a linear tale, while merge celebrates branch narratives. The decision depends on your project's needs and the story you wish to tell.

Experiment, explore, and harness the power of Git's merging artistry to sculpt your repository's history. Beyond rebase and merge, Git unveils a treasure trove of strategies, offering endless possibilities for your collaborative coding adventure.

So, venture forth armed with this knowledge, shaping your repository's saga amidst the ever-evolving landscape of team collaboration.

## Related pages

- [Git File and Commit Operations: Managing Changes Between States](./commit-operations.md)
- [Managing Local Modifications and Remote Changes in Git](./pull-changes-with-conflicts.md)
- [Understanding Git Pull vs Merge in Git Workflow](./git-pull-vs-git-merge-equivalence.md)
- [Mastering Git Branch Handling: Strategies for Deletion and Recovery](./handling-branch-deletion.md)


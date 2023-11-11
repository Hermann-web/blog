#!/bin/bash

ensure_in_correct_repo() {
    # Check if the current directory is a Git repository
    if [ ! -d ".git" ]; then
        echo "Error: Not in a Git repository."
        return 1
    fi

    # Check if the remote origin is set to the correct URL
    remote_url=$(git config --get remote.origin.url)

    if [ "$remote_url" != "https://github.com/Hermann-web/hermann-web.github.io.git" ]; then
        echo "Error: Incorrect remote origin URL. It should be https://github.com/Hermann-web/hermann-web.github.io.git"
        return 1
    fi

    echo "You are in the correct Git repository."
    return 0
}

get_to_top_level_in_repo_tree() {
    cd "$(git rev-parse --show-toplevel)"
}

function add_commit_push() {
    # Check if a branch name is provided
    if [ -z "$1" ]; then
        echo "Error: Please provide a branch name as an argument."
        return 1
    fi

    # Verify if the current branch is the provided branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [ "$current_branch" != "$1" ]; then
        echo "Error: Not in the specified branch '$1'."
        return 1
    fi

    # Check if there are changes to commit
    if [ -z "$(git status --porcelain)" ]; then
        echo "No changes to commit."
        return 0
    fi

    # Add all changes
    git add -A

    # Commit changes
    git commit -m "Commit changes"

    # Check if commit was successful
    if [ $? -ne 0 ]; then
        echo "Error: Commit failed."
        return 1
    fi

    # Determine if the "--force" option should be included in the push command
    force_option=""
    if [ "$2" == "force" ]; then
        force_option="--force"
    fi

    # Push changes to the remote repository with --set-upstream and optionally --force
    git push $force_option --set-upstream origin "$1"

    # Check if push was successful
    if [ $? -ne 0 ]; then
        echo "Error: Push failed."
        return 1
    fi

    echo "Changes successfully added, committed, and pushed to '$1'."
    return 0
}



create_new_venv() {
    # Deactivate the current environment
    deactivate

    # Remove existing virtual environment directory if it exists
    if [ -d "venv.prod" ]; then
        rm -rf venv.prod/
    fi

    # Activate the Conda environment named 'web-dev'
    conda activate web-dev

    # Create a new virtual environment named 'venv.prod'
    python -m venv venv.prod

    # Activate the new virtual environment
    source venv.prod/bin/activate
}

checkout_add_commit_push() {
    # Check out to the "source" branch
    git checkout source

    # Check if the current branch is "source"
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    
    if [ "$current_branch" != "source" ]; then
        echo "Error: Unable to switch to the 'source' branch."
        return 1
    fi

    # Add, commit, and push changes using the previous function
    add_commit_push source
}

install_requirements() {

    # Check if the directory exists
    if [ ! -d "web-source" ]; then
        echo "Error: 'web-source' directory not found."
        return 1
    fi

    # Change to the "web-source" directory
    cd web-source

    # Install requirements using pip
    pip install -r requirements-full.txt

    # Check if the installation was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install requirements."
        return 1
    fi

    echo "Requirements installed successfully."
    return 0
}

rewrite_master_branch() {
    # Check if the current branch is "source"
    current_branch=$(git rev-parse --abbrev-ref HEAD)

    if [ "$current_branch" != "source" ]; then
        # Check out to the "source" branch
        git checkout source
        
        # Check if the checkout was successful
        if [ $? -ne 0 ]; then
            echo "Error: Unable to switch to the 'source' branch."
            return 1
        fi
    fi

    # Delete the local "master" branch
    git branch -D master

    # Check if the deletion was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to delete the 'master' branch."
        return 1
    fi

    # Create and switch to a new "master" branch based on the current state of "source"
    git checkout -b master

    # Check if the new branch creation was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create and switch to the 'master' branch."
        return 1
    fi

    echo "Master branch successfully rewritten based on the 'source' branch."
    return 0
}

build_and_move_docs() {
    # Build documentation using mkdocs
    mkdocs build

    # Check if the build was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build documentation."
        return 1
    fi

    # Move the built documentation to the temporary directory
    rm -r ../tmp-dir
    mv site ../tmp-dir

    # Move back to the parent directory
    cd ..

    # Remove the original "web-source" and "web" directories
    rm -r web-source
    rm -r web

    # Move the temporary directory to the "web" directory
    mv tmp-dir web

    echo "Documentation built and moved successfully."
    return 0
}


main_fct() {
    # Step 1: Make sure I'm in the correct repository
    echo "Step 1: Making sure I'm in the correct repository..."
    ensure_in_correct_repo || return 1

    # Step 2: Change to the main project folder
    echo "Step 2: Changing to the main project folder..."
    get_to_top_level_in_repo_tree || return 1

    # Step 3: Add and commit files to the current branch
    echo "Step 3: Adding and committing files to the current branch..."
    add_commit_push source || return 1

    # Step 4: Create a Python environment for production
    echo "Step 4: Creating a Python environment for production..."
    create_new_venv || return 1

    # Step 5: Commit files to the 'source' branch
    echo "Step 5: Committing files to the 'source' branch..."
    checkout_add_commit_push || return 1

    # Step 5.5: install_requirements
    echo "Step 5.5: install_requirements..."
    install_requirements || return 1

    # Step 6: Create a new branch based on the current state of the "source" branch and then delete the "master" branch
    echo "Step 6: Creating a new branch based on the current state of the 'source' branch and then deleting the 'master' branch..."
    rewrite_master_branch || return 1

    # Step 7: Build and move documentation
    echo "Step 7: Building and moving documentation..."
    build_and_move_docs || return 1

    # Step 8: Add, commit, and push changes to the "master" branch
    echo "Step 8: Adding, committing, and pushing changes to the 'master' branch..."
    add_commit_push master force || return 1

    git checkout source

    return 0
}

main_fct

---
date: 2023-10-09
authors: [hermann-web]
description: >
  This blog shows useful stuff to know about or learn to do as web developer or data scientist/engineer
  Always working the fastest and easiest ways toward success
categories:
  - devops
  - conversion-tools
  - file-handling
  - markdown
  - beginners
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: "grip: A github-like markdown viewer in your computer"
---

## Introduction
[grip](https://github.com/joeyespo/grip) is a cli tool that you run in a directory from the terminal. 
He can parse you mardown files like github do.

??? question "But How that works and how to use it ? "

<!-- more -->

He access (endpoint based) all files from the repo and parse markdown files after sending them to github unless you use the offline renderer.

## Alternatives
- VScode Extension [markdown preview enhanced](): not working on ubuntu (see [ref to solution](https://github.com/coder/code-server/issues/4421)), i use [grip](https://github.com/joeyespo/grip)


## Prereqsites
- python: you can install the latest version

## Installation
Install it with `pip install grip`
??? info "As simple as that !"

## Usage
```bash
cd path/to/project
```
- run readme.md as web app
```bash
grip 
```
- or run another file
```bash
grip my-file-name.md
```

## Exports
### export to pdf or html
You can export you file to pdf or html. To use this feature, add `--export` option followed by the filename (with html or pdf extension)

**troubleshots**
- he can export with `grip my-file-name.md --export inut.pdf` but there is a bug 
- so i installed with `sudo apt install grip` but it takes forever
- anyway, the web view is cool. And if i neeeed pdf, from this [discussion](https://gist.github.com/justincbagley/ec0a6334cc86e854715e459349ab1446), i can use: 
    - windows: the extension `markdown preview enhanced` works there fine
    - mardown-pdf from npm
    - print the page from the webview and select non-empty pages. The pb is the ref links: the ref inside text to biblio will try to lead the grip webview that will not be open if not run on cli 

## rate limiting
`grip` have a rate limit of 60 requests/hour. So each time you save your work and the grip server is running, you lose one request. And it you use VSCode auto-save, you're kind of screwed.
Prefer either
- not to save, unless you finish modifications
- not to run grip while editing
- or add a `--norefresh` option

You can have 5000 requests/hour if you add option --user with your credentials like this
```bash
grip my_file.md --user hermann-web:<my-token>
```

## Offline renderer
!!! note "Note"
    There is an offline renderer but [it didn't make 2.0 release](https://github.com/joeyespo/grip/issues/35#issue-20152565). 

When it will be available, it quite is useful if
- you don't have internet connection
- or have have issue about sending your sensitive files to github-microsoft
You can use it like this
```bash
grip my_file.md --ofline-renderer
```

## Cool features
- add `-b` if you want it to open a browser tab for you
- `--quiet` to avoid printing in the terminal
Use `--help` to see more of this

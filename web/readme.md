---
author: Hermann Agossou
title: web
date: 2023/11/04
---

# web

## run the whole thins

### what didn't work
i'could run the app

- from `pnpm install` + `pnpm start`
    - dont serve on localhost (neither 3000 or 8000) bof and no info in it

- from `npm install -g mkdocs`
    - ùkdocs could not be found in cli
    - the moduke have been depreciated i guess

- from `npm install -g mkdoc`
    - useful for markdown processing (like to html, document python or js app from comment, ...) but not this ([docu on mkdoc](https://www.npmjs.com/package/mkdoc#doc))

- `conda install -c conda-forge mkdocs-material` ([conda](https://anaconda.org/conda-forge/mkdocs-material)) + `mkdocs serve`
    - he can't find mkdocs on cli
    - there was a problem and this [solution](https://github.com/byrnereese/mkdocs-minify-plugin/issues/2) dont match for me

- `pip install -r requirements.txt` in an 3.9 venv 
    - installation passed but `mkdocs serve` serve some bug

### what worked
it work with
- creating a env from conda, installing, 3 modules (mkdocs to [create a docu from scratch](), mkdocs-material to serve and [mkdocs-minify-plugin](https://pypi.org/project/mkdocs-minify-plugin/) to solve a bug breaking the serve) and it is okay like documented just below

```bash
# create a virtualenv (it was python 3.9.18 for the record)
conda activate web-dev
python -m venv venv
source venv/bin/activate

# install python depencencies
## simple install (the forst time)
pip instal mkdkcs mkdocs-material mkdocs-minify-plugin
## more precise (the versions installed)
pip instal mkdocs==1.5.3 mkdocs-material==9.4.7 mkdocs-minify-plugin==0.7.1
## all in (all the modules installed in the venv)
pip install requirements-full.txt

## serve app
mkdocs serve
# the app is on localhost:8000
```


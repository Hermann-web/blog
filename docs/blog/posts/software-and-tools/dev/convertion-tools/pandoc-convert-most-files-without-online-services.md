---
date: 2023-10-10
authors: [hermann-web]
comments: true
language: en
description: >
  This blog shows useful stuff to know about or learn to do as web developer or data scientist/engineer
  Always working the fastest and easiest ways toward success
categories:
  - devops
  - conversion-tools
  - file-handling
  - beginners
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: "pandoc: convert most files without online services"
---

## Introduction
Pandoc is a versatile document conversion tool that can convert Markdown documents to PDF, HTML, Word DOCX, and many other formats. Pandoc provides a wide range of options to customize the output of the converted document. Here is a list of some of the most commonly used options:

- `-s`: Create a standalone document with a header and footer.
- `-o`: Specify the output file name.
- `--from`: Specify the input format explicitly.
- `--to`: Specify the output format explicitly.

<!-- more -->

- `-V/--variable`: Set a template variable when rendering the document in standalone mode.
- `--defaults`: Specify a package of options in the form of a YAML file.
- `--list-input-formats`: Print a list of supported input formats.
- `--list-output-formats`: Print a list of supported output formats.
- `--list-highlight-styles`: Print a list of supported syntax highlighting styles.
- `-f`: Specify the input format.
- `-t`: Specify the output format.

For a full list of options, see the Pandoc User's Guide[^pandoc-cli-1] or the Pandoc manual [^pandoc-cli-5].


[^pandoc-cli-1]: [Pandoc Manual - pandoc.org](https://pandoc.org/MANUAL.html)
[^pandoc-cli-2]: [Pandoc Extras - pandoc.org](https://pandoc.org/extras.html)
[^pandoc-cli-3]: [Pandoc Filters - pandoc.org](https://pandoc.org/filters.html)
[^pandoc-cli-4]: [RMarkdown Cookbook - LaTeX Variables - bookdown.org/yihui](https://bookdown.org/yihui/rmarkdown-cookbook/latex-variables.html)
[^pandoc-cli-5]: [Pandoc General Writer Options - pandoc.org](https://pandoc.org/chunkedhtml-demo/3.3-general-writer-options.html)
[^pandoc-cli-6]: [Pandoc Specifying Formats - pandoc.org](https://pandoc.org/chunkedhtml-demo/2.2-specifying-formats.html)
[^pandoc-cli-7]: [Converting Markdown to Beautiful PDF with Pandoc - jdhao.github.io](https://jdhao.github.io/2019/05/30/markdown2pdf_pandoc/)

## Installing pandoc
- Installing Pandoc on Ubuntu
remove any previous versions of Pandoc.
```bash
sudo apt-get purge --auto-remove pandoc
```

- Download the latest version of Pandoc from the [Pandoc GitHub releases page](https://github.com/jgm/pandoc/releases)
for example,
```bash
wget https://github.com/jgm/pandoc/releases/download/3.1.8/pandoc-3.1.8-1-arm64.deb
```

- Install the downloaded package by running the command sudo dpkg -i pandoc-<version>-1-amd64.deb, replacing <version> with the version number of the package you downloaded.
for example,
```bash
dpkg -i pandoc-3.1.8-1-arm64.deb
```

- Run the following command
```bash
pandoc -f markdown -t latex input.md -o output.tex
```
where `input.md` is the name of your markdown file and `output.tex` is the name you want to give to the resulting LaTeX file.

!!! quote "[For a hard markdown user like me, pandoc is a big time relief as i can note in markdown, store mostly markdown files and still, receiving and sharing files in proprietary like pdf, docx, ...]"

### get your options
You can convert from any to any. 
To see available input (-f) format, use `pandoc --list-input-formats`
To see available output (-t) format, use `pandoc --list-output-formats`

```bash
pandoc -f markdown -t latex input.md -o output.tex
```
where `input.md` is the name of your markdown file and `output.tex` is the name you want to give to the resulting LaTeX file.

### Quick examples
- md to docx
```
pandoc my_file.md -s -t docx -o my_file.docx
```
`-s` is for standalone, `-o` to specify output file path, `-t` to specify output format (but no need as he guess from output file format given)

- md to tex
```
pandoc my_file.md -s -t latex -o my_file.tex
```

- one md to pdf
```
pandoc my_file.md -o my_file.pdf
```

- if there is an encoding problem 
```
pandoc my_file.md -o my_file.pdf --pdf-engine=lualatex
```
found in [this stackexchange discussion](https://tex.stackexchange.com/questions/685719/pandoc-latex-error-unicode-character-%E2%88%80-u2200not-set-up-for-use-with-latex)

- many md to pdf
```
pandoc *.md -o markdown_book.pdf 
```
found in [this stackoverflow discussion](https://stackoverflow.com/questions/4779582/markdown-and-including-multiple-files)

- many md to pdf accross folders
```
pandoc *.md */*.md -o markdown_book.pdf --pdf-engine=lualatex
```
note that images not accessible from the current directory will not be parsed

- add automatic section numbering like in latex
```
pandoc my_file.md -s -o my_file.pdf --number-sections
```


## Cool features in md to pdf
> from is a great blog in this [^pandoc-cli-7]
There are several cool options available when converting Markdown to PDF using Pandoc. Here are some of them:

1. `--toc`: Adds a table of contents to the beginning of the PDF that links to the various sections of the document.
2. `--template`: Allows you to use a custom LaTeX template to modify the appearance of the PDF.
3. `--variable`: Allows you to set variables that can be used in the LaTeX template. For example, you can set the font size or color of the text.
4. `--highlight-style`: Allows you to set the syntax highlighting style for code blocks.
5. `--number-sections`: Numbers the sections of the document.
6. `--metadata`: Allows you to set metadata for the PDF, such as the title, author, and date.
7. `-f markdown-implicit_figures`: [...](https://stackoverflow.com/questions/49482221/pandoc-markdown-to-pdf-image-position)

For example, here is a command that uses some of these options:

```
pandoc input.md -o output.pdf --toc --template=mytemplate.tex --variable=fontsize:12pt --highlight-style=pygments --number-sections --metadata=title:"Document" --metadata=author:"Hermann Agossou"
```

This command adds a table of contents, uses a custom LaTeX template called `mytemplate.tex`, sets the font size to 12pt, uses the Pygments syntax highlighting style, numbers the sections, and sets the title and author metadata for the PDF.

## Adding Footnote Citations in Markdown Files for Pandoc PDF Conversion

This section is about adding footnote citations using Pandoc and referencing a BibTeX file

### 1. Creating a BibTeX File

In a separate BibTeX file (e.g., `references.bib`), store your references in the BibTeX format. Here's an example:

```bibtex
@online{las-1,
  author       = "{ArcGIS}",
  title        = "{Storing lidar data}",
  howpublished = "\url{https://desktop.arcgis.com/fr/arcmap/latest/manage-data/las-dataset/storing-lidar-data.htm}",
}

@online{las-2,
  author       = "{American Society for Photogrammetry and Remote Sensing}",
  title        = "{LAS specification, version 1.4 – R13}",
  date         = "2013-07-15",
  url          = "https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf",
}
```

### 1. Formatting Citations in Markdown

To include citations in your Markdown file for conversion to PDF using Pandoc, use the `[@citation]` format within the text where you want the citation to appear. Here is an example:

```markdown
Here is a statement requiring citation [@las-2].
```

Or you can also list the references. they will be parsed as regular mardown content

```markdown
Here is a statement requiring citation [@las-2].

...

# References

[@las-1]: ArcGIS. "Storing lidar data." [Link](https://desktop.arcgis.com/fr/arcmap/latest/manage-data/las-dataset/storing-lidar-data.htm)

[@las-2]: American Society for Photogrammetry and Remote Sensing. "LAS specification, version 1.4 – R13." [Link](https://www.asprs.org/wp-content/uploads/2019/07/LAS_1_4_r15.pdf)
```

### 3. Using Pandoc for Conversion to PDF

When converting the Markdown file to PDF using Pandoc, include the following options:

- `--citeproc`: Enables citation processing.
- `--bibliography`: Specifies the path to your bibliography file.

Use the following command:

```bash
pandoc myfile.md -s --citeproc --bibliography=references.bib -o output.pdf
```

Replace `myfile.md` with the name of your Markdown file and `references.bib` with the actual name of your bibliography file.

Pandoc will process the citations marked in `[@citation]` format within your Markdown file and generate the corresponding footnotes or bibliography entries in the resulting PDF.

Remember to adjust the citation style and bibliography file as per your requirements.
https://pandoc.org/chunkedhtml-demo/8.20-citation-syntax.html

### footnote citations
You still can use `[^]` based citations.
There will appear at the end of each page, not at the end of the file.

**Examples**
```markdown
[^citation-1]: Full citation details here.
[^citation-2]: https://pandoc.org/chunkedhtml-demo/8.20-citation-syntax.html
[^citation-3]: [Full citation details here.](https://pandoc.org/chunkedhtml-demo/8.20-citation-syntax.html)
```

You can see more on citation styles with [^citation-styles]

[^citation-styles]: [more on citation styles](https://github.com/KurtPfeifle/pandoc-csl-testing)

## Resize image in markdown
For example, if you want the image to take 50% of the page wifth, use
```md
![Caption text](/path/to/image){ width=50% }
```

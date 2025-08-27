# What I learned today (@hermann-web)

This repository contains the source code for my personal website and blog, available at [hermann-web.github.io/blog/](https://hermann-web.github.io/blog/).

Meet Hermann, a pro at full-stack web development and data science, acing website building, backend systems, ML & DL (NLP and 3D vision included). These tutorials? They're my personal cheat sheets—where I stash those markdown, GitHub nuggets: syntax, functions, handy tools—stuff I use all the time. Now, I'm turning some into blog posts. Join me in simplifying our learning journeys, building a shared resource that evolves as we grow.

## Tech Stack

*   [MkDocs](https://www.mkdocs.org/)
*   [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
*   [Python](https://www.python.org/)
*   [Docker](https://www.docker.com/)

## Project Structure

*   `docs/`: Contains all the Markdown files for the website's content, including blog posts, projects, and pages.
*   `material/`: Contains overrides for the Material for MkDocs theme, allowing for customization of the website's appearance and behavior.
*   `scripts/`: Contains shell scripts for various development tasks, such as formatting code and checking for broken links.
*   `mkdocs.yml`: The configuration file for MkDocs, where the site's structure, theme, and plugins are defined.

## How to Contribute

1.  **Fork the repository.**
2.  **Clone your fork:**
    ```bash
    git clone https://github.com/your-username/blog.git
    ```
3.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
4.  **Run the development server:**
    ```bash
    mkdocs serve
    ```
5.  **Make your changes and submit a pull request.**

Alternatively, you can use Docker to run the project:

```bash
docker-compose up
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
date: 2024-01-11
authors: [hermann-web]
comments: true
description: >
  Learn how to overcome the challenge of redirecting URLs from an old GitHub Pages website to a new one, all while operating within the constraints of a static web page (HTML/CSS/JS). Discover the step-by-step process to ensure seamless redirection and address the issue of users encountering downtime.
categories:
  - Blog
  - web
  - frontend
  - github-pages
  - deployment
  - domain-migration
  - flask
  - troubleshooting
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practices/software-licenses.md
title: "Navigating Redirect Challenges With GitHub Pages: A Creative Approach to Domain Migration"
---

# Navigating Redirect Challenges With GitHub Pages: A Creative Approach to Domain Migration

## Introduction

Imagine having a GitHub Pages website. Now, you've migrated to project on another GitHub Pages website. As reports surfaced about users being unable to access the site, the need for a swift redirection from old URLs to the current ones became paramount. The catch? The solution had to operate within the constraints of a static web page, using only HTML, CSS, and JavaScript.

While conventional methods like Flask and Frozen-Flask failed, the journey led to a creative solution using HTML and JavaScript. In this blog post, I'll share the step-by-step process of how I navigated through the obstacles and achieved seamless redirection.

<!-- more -->

## The Initial Attempts: Static HTML

### Using HTML for Single Page Redirect

My first inclination was to use HTML for redirection. To redirect only one page, a simple HTML script could be used. For instance:

!!! Example  "Using HTML for Single Page Redirect"

    ```html
    <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
    <html lang="fr">
        <head>
            <title>Accueil</title>
            <script>window.location.replace("https://hermann-web.github.io/blog/")</script>
        </head>
    <body></body>
    </html>
    ```

### Static HTML Challenges

However, it quickly became apparent that defining redirects for every page statically was impractical, as I would need to create an HTML page for each endpoint. Then, I thought of using React or Flask, but they require explicitly defined routes too.

Regardless, there is a module that uses Flask to implement the first option. So, I can combine [the list of URLs](https://github.com/Hermann-web/blog/blob/gh-pages/sitemap.xml) with the [Frozen-Flask](https://frozen-flask.readthedocs.io/en/latest/) module.

The existence of this URL list proved crucial, as without it, I would have encountered additional difficulties.

## Another Way: Flask and Frozen-Flask

Next, I explored Flask and Frozen-Flask, but challenges arose when dealing with dynamic endpoints in a static context. The attempt to freeze the Flask app yielded an error, highlighting the limitations of this approach.

### First Attempts with Flask

??? Note "Prerequisites"

    Before diving into the implementation, I set up the development environment with the following commands:

    ```bash
    # Install Flask and Frozen-Flask
    python -m venv .venv
    source .venv/bin/activate
    pip install Flask Frozen-Flask

    # Download the list of endpoints
    wget https://raw.githubusercontent.com/Hermann-web/blog/gh-pages/sitemap.xml
    ```

After environment setup, I've created three python files 

- `utils.py` to parse the sitemap.xml file and extract the necessary endpoints
- `app.py` responsible for handling the redirection logic
- `main.py` convert the Flask app into a static website using `frozen-flask` module

??? Example "First Attemps with Flask"

    === ":octicons-file-code-16: `utils.py`"

        ```python
        import xml.etree.ElementTree as ET

        # Load the XML file
        xml_file_path = "sitemap.xml"
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Define the namespace
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

        WEBSITE_URL = "https://hermann-web.github.io/blog"

        def endpoint_parser(endpoint:str):
            if endpoint.startswith("/"):
                endpoint = endpoint[1:]
            if endpoint.endswith("/"):
                endpoint = endpoint[:-1]
            return endpoint

        # Extract endpoint URLs
        def get_endpoints():
            endpoints = [url_element.text for url_element in root.findall('.//ns:loc', namespace)]
            endpoints = [endpoint_parser(url.replace(WEBSITE_URL, "")) for url in endpoints]
            return endpoints
        ```

    === ":octicons-file-code-16: `app.py`"

        ```python
        from flask import Flask, redirect
        from utils import WEBSITE_URL, endpoint_parser

        app = Flask(__name__)
        target_domain = WEBSITE_URL

        assert target_domain


        @app.route('/')
        def index():
            return redirect(target_domain)

        # Create a route for redirection
        @app.route('/<path:endpoint>')
        def redirect_to_another_server(endpoint):
            endpoint = endpoint_parser(endpoint)
            print("endpoint:",endpoint)
            if endpoint in endpoints:
                target_url = f"{target_domain}/{endpoint}"
                return redirect(target_url)
            else:
                return redirect(target_domain)
        
        if __name__ == "__main__":
            app.run(debug=True)
        ```

    === ":octicons-file-code-16: `main.py`"

        ```python
        from flask_frozen import Freezer
        from app import app

        freezer = Freezer(app)

        if __name__ == '__main__':
            freezer.freeze()
        ```

Running the app went fine:

```bash
python app.py
```

```plaintext
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
```

However, running `python main.py` to freeze the Flask app using Frozen-Flask failed due to the inability to follow external redirects.

```plaintext
RuntimeError: Following external redirects is not supported.
```

This makes sense. A static webpage cannot accept dynamic endpoints. So, I should create the endpoints manually using a for-loop, hoping it will work with Flask.

So, i've tried some workarounds


## Workarounds with Flask

### Workaround 1: Manually Creating Routes

However, failed. The issue was that I should not have duplicated function names for the routes:

??? Example "Manually Creating Routes"

    ```python
    from flask import Flask, redirect
    from get_endpoints import get_endpoints, WEBSITE_URL, endpoint_parser

    app = Flask(__name__)
    target_domain = WEBSITE_URL

    assert target_domain

    endpoints = get_endpoints()

    # Create route functions for each endpoint
    for endpoint in endpoints:
        @app.route(f'/{endpoint}')
        def redirect_to_another_server():
            target_url = f"{target_domain}/{endpoint}"
            return redirect(target_url)


    if __name__ == "__main__":
        app.run(debug=True)
    ```

A first attempt to automatically generate route functions in Flask failed, as `redirect_to_another_server` was duplicated:

```plaintext
AssertionError: View function mapping is overwriting an existing endpoint function: redirect_to_another_server
```

So, that approach faced an issue with function duplication, prompting a need for a workaround.

### Workaround 2: Dynamic Function Generation

There is another solution that involves encapsulating the route functions within another function, ensuring a unique context for each endpoint.

Another solution can be to change the function name with a decorator, but it is not possible. So, I figured I can define the functions inside another function, hoping it will work.

??? Example "Dynamic Function Generation"

    ```python
    def generate_endpoint(endpoint):
        @app.route(f'/{endpoint_parser(endpoint)}')
        def dynamic_function():
            target_url = f"{target_domain}/{endpoint}"
            return redirect(target_url)

    # Create route functions for each endpoint
    for endpoint in endpoints:
        generate_endpoint(endpoint)
    ```

The attempt to generate dynamic functions also faced an issue with function duplication.

```plaintext
AssertionError: View function mapping is overwriting an existing endpoint function: dynamic_function
```

## Final Solution: Overcoming a Static Constraint

When attempting to freeze the Flask app using Frozen-Flask, a runtime error occurs due to the inability to follow external redirects. This limitation is inherent in static web pages, preventing the use of dynamic endpoints.

To work around this constraint, a custom `404.html` page is created, embedding JavaScript to correct the URL before redirecting. This clever solution ensures that even erroneous URLs lead users to the correct destination.

```html
<!-- 404.html -->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html lang="en">

<head>
    <title>Page Not Found</title>
    <script>
        // Redirect logic to correct the URL
        let new_url = "/";
        if (window.location.href.startsWith("https://hermann-web.github.io/web")) {
            new_url = window.location.href.replace("https://hermann-web.github.io/web", "/blog");
        }
        window.location.replace(new_url);
    </script>
</head>

<body></body>

</html>
```

I noticed that all erroneous URLs redirect the user to the `404.html` page. For GitHub Pages, I made the remark that even on the `404.html` page, the erroneous URLs are conserved in the browser. So, I can just correct the last endpoint `/web/*` to the correct one `/blog/*` using JavaScript.

## Conclusion

In this guide, we explored a step-by-step approach to redirecting all pages from one domain to another using Flask and Frozen-Flask. From parsing the `sitemap.xml` file to handling dynamic endpoints and overcoming static constraints, each aspect was covered in detail. The use of a custom `404.html` page with JavaScript ensures a smooth redirection experience for users, making this solution both effective and elegant.


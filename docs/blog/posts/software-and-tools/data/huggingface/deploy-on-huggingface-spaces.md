---
date: 2023-11-15
authors: [hermann-web]
comments: true
description: >
  This blog shows useful stuff to know about or learn to do as web developer or data scientist/engineer
  Always working the fastest and easiest ways toward success
categories:
  - data
  - deployment
  - frontend
  - huggingface
  - streamlit
  - beginners
links:
  - blog/posts/a-roadmap-for-web-dev.md
  - blog/posts/code-practises/software-licences.md
title: "How to deploy a Streamlit Application on Hugging Face"
---

# How to deploy a Streamlit Application on Hugging Face

In this guide, we'll walk through the steps to deploy a Streamlit app using the Hugging Face platform. For demonstration purposes, we'll create an app that utilizes the Python module Pix2Tex. Users will be able to upload an image and get the corresponding LaTeX formula along with a rendered version.

## Prerequisites

1. Create a virtual environment:
=== ":octicons-file-code-16: `For Linux & Mac`"

    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    ```

=== ":octicons-file-code-16: `For Windows`"

    ```bash
    python -m venv myenv
    ./myenv/Scripts/activate
    ```

<!-- more -->

2. Install dependencies:

    ```bash
    pip install streamlit pix2tex pillow
    ```

## Creating the Streamlit App

3. Create the `app.py` file with your Streamlit app code.

    ```python
    # app.py
    import streamlit as st
    from PIL import Image
    from pix2tex.cli import LatexOCR

    def main():
        st.title("Image to LaTeX Formula Parser")

        # Upload image through Streamlit
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

            # Perform LaTeX OCR on the uploaded image
            latex_formula = process_image(uploaded_image)

            # Display LaTeX formula
            st.subheader("LaTeX Formula:")
            st.text(latex_formula)

            # Display parsed Markdown
            parsed_md = parse_to_md(latex_formula)
            st.subheader("Parsed Markdown:")
            st.latex(f"\n{latex_formula}\n")

    def process_image(image):
        # Perform LaTeX OCR on the image
        img = Image.open(image)
        model = LatexOCR()
        latex_formula = model(img)
        return latex_formula

    def parse_to_md(latex_formula):
        # You can implement your own logic to parse LaTeX to Markdown
        # Here's a simple example for demonstration purposes
        parsed_md = f"**Parsed Formula:** *{latex_formula}*"
        return parsed_md

    if __name__ == "__main__":
        main()
    ```

4. Test the app locally:

    ```bash
    streamlit run app.py
    ```

## Deploying on Hugging Face

5. Create a new Hugging Face space:

    - Visit [Hugging Face Spaces](https://huggingface.co/new-space).
    - Name your repository and choose the Streamlit option.

    ![creating new space](./assets/creating-new-space.png)

6. Create a Hugging Face token:

    - Visit [Hugging Face Token Settings](https://huggingface.co/settings/tokens).
    - Create a new token. the format is `hf_****`
    
    ![generate new token](./assets/1-generate-token.png)
    ![generate new token](./assets/2-generate-token.png)

7. Clone the repository and push your app:
    - clone the repo
        ```bash
        git clone https://<user_name>:<token>@huggingface.co/<user_name>/<repo_name>
        cd <repo_name>
        ```
    the token is from step 6
    - add `app.py` and `requirements.txt` into the folder
    - push the changes to huggingface
        ```bash
        git add .
        git commit -m "Add application"
        git push
        ```
    - example
        ```bash
        git clone https://hermann-web:hf_****@huggingface.co/spaces/hermann-web/pix2tex 
        cd pix2tex
        mv ../app.py app.py
        pip freeze > requirements.txt
        git add .
        git commit -m "Add application"
        git push 
        ```

8. Check your deployed application online:

    - Visit Hugging Face Repository `https://huggingface.co/<user_name>/<repo_name>`.

    ![streamlit app on huggingface](./assets/streamlit-app-on-hugginface.png)

Congratulations! Your Streamlit app is now deployed on Hugging Face. You can find a live example on [https://huggingface.co/spaces/hermann-web/pix2tex](https://huggingface.co/spaces/hermann-web/pix2tex).

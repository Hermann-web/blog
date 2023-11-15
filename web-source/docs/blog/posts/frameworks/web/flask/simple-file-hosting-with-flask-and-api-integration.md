---
date: 2023-11-14
authors: [hermann-web]
description: >
  This blog shows interesting stuff to know
  It is flavored by mkdocs
categories:
  - frameworks
  - web
  - flask
links:
  - setup/setting-up-a-blog.md
  - plugins/blog.md
title: Flask based File Hosting (web app & api & python module & cli app)
---


<!-- # Simple File Hosting with Flask -->

This guide will walk you through creating a basic file hosting web application using Flask, a lightweight web framework for Python. The application will include features such as user login, file uploads, and file listing. We'll also explore adding a simple API for interacting with the application.

## Prerequistes
- python >=3.9

## Setup Environment

1. Create a `requirements.txt` file:

```plaintext
python-slugify
python-dotenv
Flask~=2.0.1
```

<!-- more -->

2. Set up a Python virtual environment:

Open your terminal and follow these steps

=== ":octicons-file-code-16: `For Linux & Mac`"

    ```shell
    cd path/to/folder

    # check the python version and localisation
    python -V
    which python

    # create the env
    python -m venv venv

    # activate the env
    ./venv/bin/activate

    # check the python version and localisation
    python -V
    which python

    # install requirements
    pip install -r requirements.txt
    ```

=== ":octicons-file-code-16: `For Windows`"

    ```bash
    cd path/to/folder

    # check the python version and localisation
    python -V
    where.exe python

    # create the env
    python -m venv venv

    # activate the env
    ./venv/Scripts/activate

    # check the python version and localisation
    python -V
    where.exe python

    # install requirements
    pip install -r requirements.txt
    ```

## Constants Configuration

Create a `constants.py` file:

```python
import os
from pathlib import Path
from dotenv import load_dotenv

UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)

DATA_FILE = Path('data.json')

DEBUG = False  # Set this to True in the development environment

# Load environment variables from .env file
load_dotenv()

if not DEBUG:
    USERNAME = os.getenv("S_USERNAME")
    PASSWORD = os.getenv("S_PASSWORD")
    FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
else:
    USERNAME = "my-username"
    PASSWORD = "my-password"
    FLASK_SECRET_KEY = 'my-secret-key'

assert USERNAME
assert PASSWORD
assert FLASK_SECRET_KEY
```

## Creating the Flask App

Create a file named `app.py` and set up the initial Flask app:

```python
from datetime import datetime, timedelta
from functools import wraps
import json
import os
import mimetypes
from slugify import slugify
from flask import Flask, render_template, request, redirect, send_from_directory
from flask import session, jsonify
import constants

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = constants.UPLOAD_FOLDER
app.config['DATA_FILE'] = constants.DATA_FILE
app.config['SECRET_KEY'] = constants.FLASK_SECRET_KEY
app.config['DEBUG'] = constants.DEBUG
app.config['USERNAME'] = constants.USERNAME
app.config['PASSWORD'] = constants.PASSWORD
```

## (Optional) Simple Hello World App

Replace the contents of `app.py` with a basic "Hello, World!" Flask app:

```python
# ... (Previous Code: Initial Setup)

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
```

Run the app using:

```bash
python app.py
```

Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to see the "Hello, World!" message.

## Adding a Web Page to List Uploaded Files

1. Create a `template` folder and download the [index.html](https://raw.githubusercontent.com/Hermann-web/simple-file-hosting-with-flask/e3807c55fa25e19b914173555929adfd2aa5567c/templates/index.html) file into it.

2. Modify the Flask app in `app.py` to include the file listing:

```python
# ... (Initial Setup)

@app.route('/')
def index():
    list_files = os.listdir(app.config['UPLOAD_FOLDER'])
    files = [(filename, "") for filename in list_files]
    return render_template('index.html', files=files)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run()
```

!!! warning "Ensure the 'uploads' folder contains some files, then run the app to see the list."

## Adding a Login Page

1. Download the [login.html](https://raw.githubusercontent.com/Hermann-web/simple-file-hosting-with-flask/e3807c55fa25e19b914173555929adfd2aa5567c/templates/login.html) file into the templates folder.

2. Add login-related functions to `app.py`:

```python
# ... (Previous code)

def validate_credentials(username, password):
    res = (username == app.config['USERNAME'] and password == app.config['PASSWORD'])
    session['logged_in'] = res
    return res

def is_logged_in():
    return 'logged_in' in session and session['logged_in']

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_logged_in():
            session['previous_url'] = request.url
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function
```
??? tip "Here is how it works"

    - `login_required` is a wrapper that use the function `is_logged_in` to check if a user is logged in

    - `validate_credentials` check if the `username` and `password` sent by the user match those we have from `constants.py`

Now, we will create the login page and add the login wrapper to the home page
```python
@app.route('/')
@login_required
def index():
    list_files = os.listdir(app.config['UPLOAD_FOLDER'])
    files = [(filename, "") for filename in list_files]
    return render_template('index.html', files=files)

@app.route('/login', methods=['GET', 'POST'])
def login():

    sucessful_login_redirect = lambda : redirect(session.pop('previous_url') if 'previous_url' in session else "\\")
    default_login_render = lambda : render_template('login.html')

    if is_logged_in():
        return sucessful_login_redirect()

    if request.method != 'POST':
        return default_login_render()

    username = request.form['username']
    password = request.form['password']

    if validate_credentials(username, password):
        return sucessful_login_redirect()
    
    return default_login_render()
```

??? tip "Here is how it works"
    - The login page use the template `index.html`, a form with two fields: `username` and `password`.
    But if he is already logged in, he may be redirected to another page.
    - When he add his credentials and send them, we will get `username` and `password` from the form.
    - Then we will use the function `validate_credentials` to check them.
    - If the credentials match, the user is redirected to the page he asks for. For example, in an unauthenticated user go the the home page, he is redirected to the login page. And if his crede,tials match, he is redirected back to the home page.

## create your credentials to access the app
As i've 've said, i use the most basic authentication for this simple login protected web app.To add the credentials for the web app, create a file `.env` like [.env.example](https://raw.githubusercontent.com/Hermann-web/simple-file-hosting-with-flask/e3807c55fa25e19b914173555929adfd2aa5567c/.env.example)

`.env`
```.env
USERNAME=myuser
PASSWORD=mypassword
```
!!! warning
    Modify it to match the credentials for your app

    Also, go into the constants.py file and make sure `DEBUG` is set to `False` to use it


## (Optional) Session Timeout and Logout Page
You can add a timeout of the session. So a user will not stay logged in forever. It is important for data sentivive related apps.
You can also let the user logout if he wants.
There is a logout bouton in the home page.

1. Set the session timeout:

```python
# ... (Previous code)

# Set the session timeout to 30 minutes (1800 seconds)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
```

2. Add a logout page:

```python
# ... (Previous code)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')
```

## Adding File Download and Upload Features


1. Add functions for file download:

```python
# ... (Previous code)

@app.route('/uploads/<path:filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
```

2. Add functions for file upload:

```python
# ... (Previous code)

@app.route('/uploads/<path:filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def slugify_filename(filename):
    # Split the filename and extension
    _ = filename.rsplit('.', 1)
    if len(_)<2: return 
    base, extension = _
    # Slugify the base part
    slug_base = slugify(base)
    # Join the slugified base with the original extension
    slug_filename = f"{slug_base}.{extension}"
    return slug_filename

def handle_file_saving(file):
    filename = slugify_filename(file.filename)
    file_save = app.config['UPLOAD_FOLDER'] / filename
    print(f"saving {file_save.resolve()}")
    file.save(file_save)
    return filename

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    file = request.files['file']
    if file:
        filename = handle_file_saving(file)
    return redirect('/')
```

??? tip "Here is how it works"
    We have added a upload endpoint 
    - that will receive user files and save them using the `handle_file_saving` function
    - that is protected with `login_required`
    - The function `slugify_filename` will rewrite the filename to use only lowercase alphanumeric characters an `-` as separators instead of space
    - `handle_file_saving` will save the file in the `uploads` directory

The complete code with file download and upload features can be found in the [GitHub repository](https://github.com/Hermann-web/simple-file-hosting-with-flask).

## (Optional) Adding Endpoints for Open File, Raw Content, and API

1. Add endpoints for opening a file, displaying raw content, and API:

```python
# ... (Previous code)

@app.route('/open/<path:filename>')
@login_required
def open_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(file_path):
        return "File not found"

    mime_type = get_content_type(file_path)

    # Map .md and .mmd extensions to text/plain
    if mime_type == 'text/markdown' or mime_type == 'text/x-markdown':
        mime_type = 'text/plain'

    if mime_type:
        with open(file_path, 'rb') as file:
            file_content = file.read()
        return Response(file_content, content_type=mime_type)

    return "Unknown file type"

@app.route('/raw/<path:filename>')
@login_required
def raw_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(file_path):
        return "File not found"

    with open(file_path, 'rb') as file:
        file_content = file.read()
    return file_content
```

The code for these features is available in the [GitHub repository](https://github.com/Hermann-web/simple-file-hosting-with-flask).

## (Optional) Modifying File Upload to Filter Files

To filter the files, you can use a database to add which files to show. To be simple, i've used a json file.

So the
- home page will look into the json file to show the files
- the upload page will save the file on the server and also add it in the json file


1. Modify the listing feature to filter files using a JSON file:

```python
def load_data_from_json():
    if os.path.exists(app.config['DATA_FILE']):
        with open(app.config['DATA_FILE'], 'r') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                pass
    return {}

def get_files_with_dates():
    data = load_data_from_json()
    return [(filename, data[filename]) for filename in sorted(data, key=data.get) if (app.config['UPLOAD_FOLDER']/filename).exists()]

@app.route('/')
@login_required
def index():
    files = get_files_with_dates()
    return render_template('index.html', files=files)
```

2. Modify the upload feature to filter files using a JSON file:

```python
def update_data_file(filename):
    data = load_data_from_json()
    data[filename] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(app.config['DATA_FILE'], 'w') as file:
        json.dump(data, file)

def handle_file_saving(file):
    filename = slugify_filename(file.filename)
    file_save = app.config['UPLOAD_FOLDER'] / filename
    print(f"saving {file_save.resolve()}")
    file.save(file_save)
    update_data_file(filename)
    return filename

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    file = request.files['file']
    if file:
        filename = handle_file_saving(file)
    return redirect('/')
```

The complete code with file filtering and other features is available in the [GitHub repository](https://github.com/Hermann-web/simple-file-hosting-with-flask).


## (Optional) Adding an api along the web page
1. login
```python
@app.route('/api/login', methods=['POST'])
def api_login():
    username = request.json.get('username')
    password = request.json.get('password')

    if validate_credentials(username, password):
        return jsonify({'message': 'Login successful'})
    else:
        return jsonify({'message': 'Invalid credentials'}), 401
```

2. get all the files
```python
@app.route('/api')
def api_index():
    if not is_logged_in():
        return jsonify({'message': 'Unauthorized'}), 401

    files = get_files()
    return jsonify({'files': files})
```

3. upload a file
```python
@app.route('/api/upload', methods=['POST'])
def api_upload():
    if not is_logged_in():
        return jsonify({'message': 'Unauthorized'}), 401

    file = request.files['file']
    if file:
        filename = handle_file_saving(file)
        return jsonify({'message': f'File uploaded: {filename}'})
    else:
        return jsonify({'message': 'No file provided'}), 400
```

4. download a file
```python
@app.route('/api/uploads/<path:filename>')
def api_download(filename):
    if not is_logged_in():
        return jsonify({'message': 'Unauthorized'}), 401

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
```

## (Bonus) How to use the api
- you can access the api with the routes `http://localhost:5000/api/*`
-  The file [cli_app/cli_app.py](https://github.com/Hermann-web/simple-file-hosting-with-flask/blob/e3807c55fa25e19b914173555929adfd2aa5567c/cli_app/cli_app.py) to access the api along with a context manager to handle sessions
- you can read the [api documentation](https://github.com/Hermann-web/simple-file-hosting-with-flask/blob/e3807c55fa25e19b914173555929adfd2aa5567c/docs/api.md)

## (Bonus) How to use the cli app
- The script [cli_app/sharefile.py](https://github.com/Hermann-web/simple-file-hosting-with-flask/blob/e3807c55fa25e19b914173555929adfd2aa5567c/cli_app/sharefile.py) provides a cli app to access the api context manager
- Using your cli, you can list, upload and download files. The api will be called behind the hood by [cli_app/cli_app.py](https://github.com/Hermann-web/simple-file-hosting-with-flask/blob/e3807c55fa25e19b914173555929adfd2aa5567c/cli_app/cli_app.py)
- you can read the [cli-app documentation](https://github.com/Hermann-web/simple-file-hosting-with-flask/blob/e3807c55fa25e19b914173555929adfd2aa5567c/docs/cli-app.md)

## (Bonus) Serving Static Files

If you want to serve static files, add the following endpoint:

```python
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)
```

Create a 'static' folder and place your static files inside it.

It can be interesting for custom css/js files and others

You can find all this code in the repository [https://github.com/Hermann-web/simple-file-hosting-with-flask](https://github.com/Hermann-web/simple-file-hosting-with-flask)


from setuptools import setup, find_packages

setup(
    name='mkdocs-blog-plugin',
    version='0.1',
    description='A custom blog plugin for MkDocs',
    packages=find_packages(),
    entry_points={
        'mkdocs.plugins': [
            'blog = plugins.blog.plugin:BlogPlugin',
        ]
    },
)

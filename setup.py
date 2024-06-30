### Arquivo `setup.py`

from setuptools import setup, find_packages

setup(
    name="markov_points",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
    ],
    author=["Ataíde Gualberto", "Jugurta Montalvão", "Gabriel Bastos", "Rodrigo Passos"],
    author_email="ataidegualberto.eng@gmail.com",
    description="A package to generate state-embeddings from markov chains.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ataidegualberto/Markov_Points",
    python_requires='>=3.7',
    license='MIT',
)

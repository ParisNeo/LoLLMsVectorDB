from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read the requirements from the requirements.txt file
requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name='lollmsvectordb',
    version='1.1.5',
    description='A modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='ParisNeo',
    author_email='parisneoai@gmail.com',
    url='https://github.com/ParisNeo/LoLLMsVectorDB',  # Replace with the actual URL of your repository
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

from setuptools import setup, find_packages

setup(
    name='lollmsvectordb',
    version='0.1.0',
    description='A modular text-based database manager for retrieval-augmented generation (RAG), seamlessly integrating with the LoLLMs ecosystem.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='ParisNeo',
    author_email='parisneoai@gmail.com',
    url='https://github.com/ParisNeo/LoLLMsVectorDB',  # Replace with the actual URL of your repository
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'sqlite3'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

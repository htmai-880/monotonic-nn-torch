from setuptools import setup, find_packages

setup(
    name='monotonic-nn-torch',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'scikit-learn',
        'pandas'
    ],
    author='Huu Tan Mai',
    author_email='huutan.mai@gmail.com',
    description='A package for monotonic neural networks using PyTorch',
    url='https://github.com/yourusername/monotonic-nn-torch',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
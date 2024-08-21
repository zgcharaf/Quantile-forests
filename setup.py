from setuptools import setup, find_packages

setup(
    name="quantile_forest",
    version="0.1.0",
    description="A library for applying quantile regression techniques to tree-based models using Random Forest Proximities.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Your Name",
    author_email="zgcharaf@gmail.com",
    url="https://github.com/zgcharaf/Quantile-forests",  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "joblib",
        "optuna",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

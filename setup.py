import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mousetracker",
    version="0.0.1",
    author="Mira Davidson",
    author_email="msd115@ic.ac.uk",
    description="Pose estimation and feature extraction for analysis of mouse behaviour",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='3.6',
)

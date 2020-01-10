import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="harvesters-simple-capture-gui",
    version="0.0.1",
    author="Brian Ofrim",
    author_email="bofrim@ualberta.ca",
    description="A simple image acquisition, display and saving tool build with harvesters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BrianOfrim/harvesters-simple-capture-gui",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu",
    ],
    python_requires=">=3.6",
)

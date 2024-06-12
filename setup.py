import setuptools

with open("./README.md", encoding="utf-8") as f:
    long_description = f.read()
with open("./src/pytregex/about.py", encoding="utf-8") as f:
    about = {}
    exec(f.read(), about)
setuptools.setup(
    name="pytregex",
    version=about["__version__"],
    author="Long Tan",
    author_email="tanloong@foxmail.com",
    url="https://github.com/tanloong/pytregex",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    description="Python implementation of Stanford Tregex",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.7",
    entry_points={"console_scripts": ["pytregex = pytregex:main"]},
)

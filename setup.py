#!/usr/bin/env python
import os
from setuptools import find_packages, setup, Command
import codecs
import sys
from shutil import rmtree

here = os.path.abspath(os.path.dirname(__file__))

about = {}

with open(os.path.join(here, "exp", "__version__.py")) as f:
    exec(f.read(), about)

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist bdist_wheel upload")
    sys.exit()

required = [
    'toml',
    'click',
    'numpy',
    'matplotlib'
]


class UploadCommand(Command):
    """Support setup.py publish."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except FileNotFoundError:
            pass
        self.status("Building Source distribution…")
        os.system("{0} setup.py sdist bdist_wheel".format(sys.executable))
        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")
        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")
        sys.exit()

setup(name='experiment',
      version=about["__version__"],
      description='Python tool do design and run experiments with global optimisation and grid search',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Davide Nunes',
      author_email='mail@davidenunes.com',
      url='https://github.com/davidenunes/exp',
      packages=find_packages(exclude=["tests", "tests.*"]),
      install_requires=required,
      python_requires=">=3.6",
      license="Apache 2.0",
      package_data={
          "": ["LICENSE"],
      },
      include_package_data=True,
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      cmdclass={"upload": UploadCommand},
      )

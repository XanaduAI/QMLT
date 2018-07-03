# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
import sys
import os
from setuptools import setup
# from sphinx.setup_command import BuildDoc

with open("qmlt/_version.py") as f:
	version = f.readlines()[-1].split()[-1].strip("\"'")

# cmdclass = {'build_docs': BuildDoc}

# add any major python requirements
requirements = [
    "scikit-learn>=0.19",
    "matplotlib>=2.0",
    "strawberryfields>=0.7.3"
]

info = {
    'name': 'qmlt',
    'version': version,
    'maintainer': 'Xanadu Inc.',
    'maintainer_email': 'maria@xanadu.ai',
    'url': 'http://xanadu.ai',
    'license': 'LICENSE',
    # put submodules here
    'packages': [
                    'qmlt',
                    'qmlt.numerical',
                    'qmlt.tf'
                ],
    'include_package_data': True,
    'description': 'Machine learning and optimization of quantum optical circuits',
    'long_description': open('README.rst').read(),
    'provides': ["qmlt"],
    'install_requires': requirements,
    'command_options': {
        'build_sphinx': {
            'version': ('setup.py', version),
            'release': ('setup.py', version)}}
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]

setup(classifiers=classifiers, **(info))

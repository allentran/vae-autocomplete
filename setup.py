__author__ = 'allentran'

from setuptools import find_packages, setup

if __name__ == '__main__':
    name = 'vae_seq'
    setup(
        name=name,
        version="0.0.0",
        author='Allen Tran',
        author_email='realallentran@gmail.com',
        description='Complete short sequences',
        packages=find_packages(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Programming Language :: Python',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],
        setup_requires=[
            'setuptools>=3.4.4',
        ],
    )

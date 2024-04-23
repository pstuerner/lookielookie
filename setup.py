from setuptools import setup, find_packages


setup(
    name="lookielookie",
    version="1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'lookielookie = lookielookie.__main__:main',
        ],
    },
)
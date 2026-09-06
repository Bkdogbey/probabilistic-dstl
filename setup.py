from setuptools import find_packages, setup

setup(
    name='pdstl',
    package_dir={'': 'src'},
    packages=find_packages(where='src', include=['pdstl', 'pdstl.*']),
    version='0.2.0',
    description='Interval-valued temporal scores over predicate probabilities',
    python_requires='>=3.10',
    install_requires=['torch>=2.0'],
    extras_require={
        'test': ['pytest>=7'],
        'examples': ['numpy', 'scipy', 'matplotlib', 'pyyaml', 'python-dotenv'],
    },
    license='MIT',
)

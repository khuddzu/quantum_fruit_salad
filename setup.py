from setuptools import setup, find_packages

VERSION = '0.0.1'

DESCRIPTION = 'This repo, it is kind of a cosmic gumbo.'

LONG_DESCRIPTION = "Random, useful code from my own personal collection."

setup(
        name = "quantum_fruit_salad",
        version = VERSION,
        author = "Kate Huddleston",
        author_email = "khuddzu724@gmail.com",
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        packages = find_packages(),
        install_requires=['torch',
            'tqdm',],

        )


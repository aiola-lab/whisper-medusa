from setuptools import find_packages, setup


def requirements(name):
    list_requirements = []
    with open(f"{name}.txt") as f:
        for line in f:
            list_requirements.append(line.rstrip())
    return list_requirements


setup(
    name="medusa-whisper",
    packages=find_packages(),
    version="0.1.0",
    description="Adding Medusa speculative decoding to Whisper model",
    author="aiOla",
    python_requires=">=3.9",
    install_requires=requirements("requirements"),  # Optional
)

from setuptools import setup, find_packages

setup(
    name="selfperm-monitor",
    version="0.1.0",
    description="A chain-of-thought safety monitor that catches manipulation hiding behind self-permission reasoning.",
    packages=find_packages(),
    include_package_data=True,
    package_data={"selfperm_monitor": ["data/*.csv"]},
    install_requires=["groq>=0.11.0"],
    python_requires=">=3.9",
)

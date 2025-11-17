from setuptools import setup
import tomli  # or tomllib in Python 3.11+

if __name__ == "__main__":
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    
    setup(
        name=pyproject["project"]["name"],
        version=pyproject["project"]["version"],
    )
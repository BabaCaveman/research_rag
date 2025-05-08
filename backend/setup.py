"""
Setup script for the crop reflectance analysis backend
"""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="crop_reflectance_analysis",
    version="0.1.0",
    description="Backend for Crop Reflectance Analysis Web Application",
    author="Research Team",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
)
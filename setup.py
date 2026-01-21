from setuptools import setup, find_packages

setup(
    name="grounded_planner",
    version="0.1.0",
    description="Neuro-Symbolic Vision-Language-Action Pipeline for Robotic Manipulation",
    author="Aboo Bakr Muhammad",
    author_email="aboobakr9.ab@gmail.com",
    url="https://github.com/aboobakr09/grounded_vlm_planner",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "google-generativeai>=0.3.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.20.0",
        "pillow>=9.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "pybullet>=3.2.0",
            "matplotlib>=3.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Embodied AI :: Neuro-Symbolic AI :: Artificial Intelligence",
    ],
)

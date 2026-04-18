from setuptools import setup, find_packages

setup(
    name="llm-inference-engine",
    version="1.0.0",
    description="Production-grade LLM inference engine for Phi-3 Mini with hardware benchmarking",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/llm-inference-engine",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.29.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.6.0",
        "pyyaml>=6.0.1",
        "mlflow>=2.12.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "quantization": ["bitsandbytes>=0.43.0", "optimum>=1.19.0"],
        "dev": ["pytest>=8.1.0", "black>=24.3.0", "flake8>=7.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

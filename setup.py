from setuptools import setup, find_packages

setup(
    name="safelora-quantized",
    version="0.1.0",
    description="SafeLoRA for Quantized Models - ECE 285 Project",
    author="Raghu",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.14.0",
        "evaluate>=0.4.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "gptq": ["auto-gptq>=0.5.0"],
        "awq": ["autoawq>=0.1.8"],
        "eval": ["openai>=1.0.0", "rouge-score>=0.1.2"],
        "dev": ["jupyter>=1.0.0", "matplotlib>=3.7.0", "seaborn>=0.12.0"],
    },
)


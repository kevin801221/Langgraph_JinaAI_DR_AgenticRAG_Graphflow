from setuptools import setup, find_packages

setup(
    name="jina-langgraph-research-assistant",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langgraph>=0.3.0",
        "httpx>=0.28.0",
        "pydantic>=2.0.0",
        "qdrant-client>=1.13.0",
        "graphviz>=0.20.0"
    ],
    author="Kevin Luo",
    author_email="your.email@example.com",
    description="智能研究助手系統，使用 LangGraph 和 Jina AI 構建",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/jina-langgraph-research-assistant",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

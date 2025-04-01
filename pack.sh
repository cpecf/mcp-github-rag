rm -rf mcp-github-rag.zip
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
zip -r mcp-github-rag.zip src Dockerfile pyproject.toml config.json
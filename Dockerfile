from nikolasigmoid/py-mcp-proxy:latest

copy pyproject.toml pyproject.toml
copy src src
copy config.json config.json

run pip install . && rm -rf pyproject.toml
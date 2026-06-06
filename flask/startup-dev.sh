#!/bin/bash
cd /flask
uv run flask run --with-threads --debugger --host=0.0.0.0 --port=5000

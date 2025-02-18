#!/bin/bash

# Open first terminal and run the backend
gnome-terminal -- bash -c "source venv/bin/activate && uvicorn main:app --reload; exec bash"

# Open second terminal and run the frontend
gnome-terminal -- bash -c "npx vite; exec bash"


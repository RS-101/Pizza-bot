# Pizza-bor

A simple chat bot project that combines a Python backend (using FastAPI) with a TypeScript/Vite frontend. The project also integrates with the Gemini API for generative language features.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Running the Project](#running-the-project)
- [Usage](#usage)
- [License](#license)

---

## Overview

This project provides a pizza chat bot:
- **FastAPI** for the backend, with support from libraries such as `pydantic`, `ollama`, and `httpx`.
- **TypeScript and Vite** for a modern, fast frontend development experience.
- Integration with the **Gemini API** for generating content. The API key is securely managed via environment variables.

---

## Features

- **Python Backend**:
  - Built with FastAPI for creating robust REST APIs.
  - Utilizes `pydantic` for data validation.
  - Cross-Origin Resource Sharing (CORS) enabled.
  - Integrates with external APIs such as Gemini and Ollama.
  
- **TypeScript/Vite Frontend**:
  - Fast development with Vite.
  - Uses modern JavaScript/TypeScript features.
  - Axios is included for HTTP requests.
  
- **Bash Script for Easy Startup**:
  - `run_project.sh` script opens separate terminals to run the backend and frontend concurrently.

---

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.8+** (with virtual environment support)
- **Node.js and npm**
- **Vite** (installed via npm)
- **gnome-terminal** (or adjust the script for your terminal emulator)
  
---

## Installation

### 1. Clone the Repository:

```sh
git clone https://github.com/yourusername/chat-bot-template.git
cd chat-bot-template
```

### 2. Set Up the Python Backend:

Create and activate a virtual environment:

```sh
python3 -m venv venv
source venv/bin/activate
```

Install the required Python libraries:

```sh
pip install fastapi uvicorn pydantic fastapi[all] ollama httpx
```

### 3. Set Up the Frontend:

Install the Node dependencies:

```sh
npm install
```

---

## Configuration

### Gemini API Key

The project requires a Gemini API key. Set the API key as an environment variable before running the project. For example, you can add the following line to your shell configuration (e.g., `.bashrc` or `.zshrc`):

```sh
export GEMINI_API_KEY="your_gemini_api_key_here"
```

In the Python code, the API URL is configured as follows:

```python
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
```

Make sure the environment variable is set in your active session, especially within the virtual environment.

---

## File Structure

```
chat-bot-template/
├── index.html
├── package.json
├── package-lock.json
├── tsconfig.json
├── main.py
├── run_project.sh
├── public/
└── src/
    ├── main.ts
    ├── send.ts
    ├── style.css
    ├── typescript.svg
    └── vite-env.d.ts
```

- **`index.html`**: The main HTML file for the frontend.
- **`main.py`**: The entry point for the FastAPI backend.
- **`run_project.sh`**: Bash script to launch both backend and frontend.
- **`public/`**: Static assets for the frontend.
- **`src/`**: Source files for the frontend, including TypeScript and CSS.

---

## Running the Project

You can run the project using the provided bash script:

1. **Make the script executable (if not already):**

   ```sh
   chmod +x run_project.sh
   ```

2. **Run the project:**

   ```sh
   ./run_project.sh
   ```

This script will:

- Open a new terminal to activate the Python virtual environment and run the backend with Uvicorn.
- Open another terminal to start the frontend using Vite.

> **Note:** If you are using a different terminal emulator, you may need to adjust the commands in `run_project.sh` accordingly.

---

## Usage

- **Backend API**: Accessible via FastAPI. With the `--reload` flag, changes in your backend code will automatically restart the server.
- **Frontend**: Served by Vite. Open the browser at the address shown in the terminal (usually `http://localhost:3000`).

Explore and modify the code in `src/` and `main.py` to tailor the functionality of your chat bot.

---

## License

MIT

import os

# every setting can be overridden with an environment variable (or a .env file
# loaded by your shell / process manager)

# where the SQLite database lives
DB_PATH = os.getenv("AIHOST_DB_PATH", "aihost.db")

# the llama.cpp server (llama-server) that actually runs the model
LLAMA_HOST = os.getenv("AIHOST_LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.getenv("AIHOST_LLAMA_PORT", "8080"))
LLAMA_SERVER_URL = os.getenv("AIHOST_LLAMA_URL", f"http://{LLAMA_HOST}:{LLAMA_PORT}")

# name / path of the llama-server binary
LLAMA_SERVER_BIN = os.getenv("AIHOST_LLAMA_BIN", "llama-server")

# seconds to wait for the model server before giving up
REQUEST_TIMEOUT = float(os.getenv("AIHOST_REQUEST_TIMEOUT", "300"))

# where the public API (this app) listens
API_HOST = os.getenv("AIHOST_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("AIHOST_API_PORT", "8000"))

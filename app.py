from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Sample Web App</title>
        <style>
            body { font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #f4f4f9; }
            .card { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
            button { padding: 0.5rem 1rem; border: none; border-radius: 4px; background: #007bff; color: white; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Flask Web App</h1>
            <p>Your local server is up and running!</p>
            <button onclick="alert('Hello from the frontend!')">Click Me</button>
        </div>
    </body>
    </html>
    """


@app.route("/api/status")
def status():
    return jsonify({"status": "running", "environment": "development"})


if __name__ == "__main__":
    # Runs local server on port 5000 with auto-reload enabled
    app.run(host="127.0.0.1", port=5000, debug=True)
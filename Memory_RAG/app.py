from flask import Flask, request, jsonify, render_template, send_from_directory
from Cognitron import process_message_with_memory, initialize_chat_model
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

app = Flask(__name__, static_folder='.')
executor = ThreadPoolExecutor(max_workers=3)
loop = asyncio.new_event_loop()
chat_model = initialize_chat_model()  # Get the initialized model

def run_async(coro):
    """Helper function to run async code from sync context"""
    return asyncio.run_coroutine_threadsafe(coro, loop)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"reply": "Please provide a message."})

    try:
        # Run the async processing in the background
        future = executor.submit(
            lambda: asyncio.run(process_message_with_memory(chat_model, user_message))
        )
        response = future.result()
        return jsonify({"reply": response})
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({"reply": "I apologize, but I encountered an error processing your message."})

# Serve static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

def run_async_loop():
    """Run the async loop in a separate thread"""
    asyncio.set_event_loop(loop)
    loop.run_forever()

if __name__ == "__main__":
    # Start the async loop in a separate thread
    executor.submit(run_async_loop)
    app.run(debug=True, use_reloader=False)  # disable reloader to avoid duplicate loops

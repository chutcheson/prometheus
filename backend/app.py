from flask import Flask, request, jsonify, send_from_directory
from game_loop import TerminalGame
import os
import traceback

app = Flask(__name__, static_folder='../frontend', static_url_path='')
game = TerminalGame()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/execute', methods=['POST'])
def execute_command():
    try:
        data = request.json
        command = data['command']
        
        result = game.handle_command(command)
        
        return jsonify(result)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return jsonify({"error": error_message}), 500

@app.route('/initial-message')
def get_initial_message():
    try:
        result = game.handle_command('')  # Empty command to trigger initial message
        return jsonify(result)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)

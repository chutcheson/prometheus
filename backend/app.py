from flask import Flask, request, jsonify, send_from_directory
from game_loop import TerminalGame
import os

app = Flask(__name__, static_folder='../frontend', static_url_path='')
game = TerminalGame()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/execute', methods=['POST'])
def execute_command():
    data = request.json
    command = data['command']
    
    result = game.handle_command(command)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

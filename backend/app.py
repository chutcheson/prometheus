from flask import Flask, request, jsonify
from game_loop import TerminalGame

app = Flask(__name__)
game = TerminalGame()

@app.route('/execute', methods=['POST'])
def execute_command():
    data = request.json
    command = data['command']
    
    # Execute the command using your existing game logic
    result = game.handle_command(command)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

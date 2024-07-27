# Anthropic AI Research Terminal Simulator

## Overview

The Anthropic AI Research Terminal Simulator is an educational game designed for students to experience what it's like to work in a cutting-edge AI research lab. Set in a fictional Anthropic research facility in 2027, after the discovery of AGI, this simulator provides a safe and engaging environment for students to explore the social, financial, and technical aspects of AI research.

## Features

- Simulated terminal environment with both standard Unix/Linux commands and custom AI research-related commands
- Dynamic narrative that responds to user actions and progresses the story
- Challenges and missions related to advanced deep learning concepts
- AI-assisted command handling and response generation
- Split-screen interface with terminal and narrative sections

## Installation

### Prerequisites

- Python 3.7+
- Flask
- OpenAI Python library

### Steps

1. Clone the repository:

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Running the Game

1. Start the backend server:
   ```
   python backend/app.py
   ```

2. Open `frontend/index.html` in your web browser.

## How to Play

1. When you start the game, you'll see a welcome message and your first mission.
2. Use standard terminal commands (like `ls`, `cd`, `cat`) to navigate the virtual file system and interact with files.
3. Use custom commands (like `msgrcv` and `msgsnd`) to receive and respond to missions.
4. Type `help` to see available commands and get assistance.
5. Progress through the narrative by completing missions and exploring the simulated Anthropic research environment.

## Custom Commands

- `msgrcv`: Receive new missions or reminders of current missions
- `msgsnd <your_answer>`: Submit your answer or solution to the current mission
- `assistant`: Get a cryptic hint from the AI assistant
- `clear`: Clear the terminal screen

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Anthropic for inspiration and the fictional setting
- OpenAI for providing the GPT model used in the backend

## Disclaimer

This is a fictional educational game and is not affiliated with or endorsed by Anthropic. All scenarios, characters, and events are fictional and any resemblance to real persons or events is purely coincidental.

(Claude wanted that disclaimer lol)

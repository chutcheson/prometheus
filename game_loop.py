import json
from typing import Dict, Any, Optional, List
from llm_interface import LLMInterface

class FileSystem:
    def __init__(self):
        self.tree = {"/": {}}  # Root directory

    def add_file(self, path: str, content: str):
        parts = path.strip("/").split("/")
        current = self.tree["/"]
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = content

    def get_file(self, path: str) -> Optional[str]:
        parts = path.strip("/").split("/")
        current = self.tree["/"]
        for part in parts:
            if part in current:
                current = current[part]
            else:
                return None
        return current if isinstance(current, str) else None

    def get_subtree(self, path: str, depth: int = 1) -> Dict[str, Any]:
        parts = path.strip("/").split("/")
        current = self.tree["/"]
        for part in parts:
            if part in current:
                current = current[part]
            else:
                return {}

        def limit_depth(node, current_depth):
            if current_depth == depth:
                return {k: "..." for k in node if isinstance(node[k], dict)}
            return {k: limit_depth(v, current_depth + 1) if isinstance(v, dict) else v 
                    for k, v in node.items()}
        
        return limit_depth(current, 0)

class GameState:
    def __init__(self):
        self.file_system = FileSystem()
        self.current_directory = "/"
        self.env_vars = {}
        self.current_mission: Optional[str] = None
        self.mission_progress: Dict[str, Any] = {}
        self.narrative_history: List[str] = []
        self.custom_commands: Dict[str, str] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_system": self.file_system.get_subtree(self.current_directory, depth=2),
            "current_directory": self.current_directory,
            "env_vars": self.env_vars,
            "current_mission": self.current_mission,
            "mission_progress": self.mission_progress,
            "narrative_history": self.narrative_history[-5:],  # Last 5 narrative events
            "custom_commands": self.custom_commands
        }

class TerminalGame:
    def __init__(self):
        self.state = GameState()
        self.llm = LLMInterface()
        self.missions = self._load_missions()

    def _load_missions(self) -> List[Dict[str, str]]:
        return [
            {"question": "What is the fundamental unit of deep learning models?", "answer": "neuron"},
            {"question": "What phenomenon occurs when an LLM generates false or nonsensical information?", "answer": "hallucination"},
            {"question": "What is the name of the attention mechanism used in transformer models?", "answer": "self-attention"},
            {"question": "What type of neural network is particularly effective for processing sequential data?", "answer": "recurrent neural network"},
            {"question": "What technique is used to prevent overfitting by randomly dropping out neurons during training?", "answer": "dropout"},
        ]

    def run(self):
        print("Welcome to the Anthropic AI Research Terminal (2027 Edition). Type 'help' for available commands.")
        while True:
            user_input = input(f"{self.state.current_directory}$ ")
            command, *args = user_input.split()

            if command == "exit":
                print("Thank you for using the Anthropic AI Research Terminal. Goodbye!")
                break
            elif command == "msgrcv":
                self._handle_msgrcv()
            elif command == "msgsnd":
                self._handle_msgsnd(args)
            elif command == "help":
                self._handle_help(args)
            else:
                self._handle_regular_command(command, args)

    def _handle_msgrcv(self):
        prompt = f"""
        The user has requested to receive a message (msgrcv command) in the Anthropic AI Research Terminal.
        Current game state:
        {json.dumps(self.state.to_dict(), indent=2)}

        Please provide a response that:
        1. If there's a current mission, reminds the user of the mission details.
        2. If there's no current mission, generates a new mission related to advanced deep learning concepts.
        3. Includes some narrative flavor text about the AI-managed system.

        Respond with a JSON object in the following format:
        {{
            "mission": "Mission details or new mission",
            "narrative": "Narrative flavor text"
        }}

        Example response:
        {{
            "mission": "Explain the concept of attention mechanisms in transformer models",
            "narrative": "The AI-managed system hums with an otherworldly energy. As you interface with the terminal, you feel a slight tingling sensation in your fingertips, as if the very fabric of reality is being manipulated by the AGI's computations."
        }}

        Please provide your response in valid JSON format:
        """
        response = self.llm.query_json(prompt)
        
        print(response["narrative"])
        if self.state.current_mission:
            print(f"Current mission: {response['mission']}")
        else:
            self.state.current_mission = response['mission']
            print(f"New mission received: {response['mission']}")

    def _handle_msgsnd(self, args):
        if not args:
            print("Usage: msgsnd <your_answer>")
            return

        answer = " ".join(args)
        prompt = f"""
        The user has submitted an answer (msgsnd command) in the Anthropic AI Research Terminal.
        Current mission: {self.state.current_mission}
        User's answer: {answer}

        Please evaluate the answer and provide a response that:
        1. Determines if the answer is correct.
        2. Provides feedback on the answer.
        3. If correct, generates a new mission related to advanced deep learning concepts.
        4. Includes some narrative flavor text about the AI-managed system.

        Respond with a JSON object in the following format:
        {{
            "correct": true/false,
            "feedback": "Feedback on the answer",
            "new_mission": "New mission if the answer was correct, or null",
            "narrative": "Narrative flavor text"
        }}

        Example response:
        {{
            "correct": true,
            "feedback": "Excellent explanation! You've demonstrated a clear understanding of attention mechanisms in transformer models.",
            "new_mission": "Describe the role of layer normalization in deep neural networks",
            "narrative": "As your answer resonates through the system, you notice a subtle shift in the air around you. The AI seems to acknowledge your understanding, and the terminal's glow intensifies momentarily."
        }}

        Please provide your response in valid JSON format:
        """
        response = self.llm.query_json(prompt)
        
        print(response["narrative"])
        print(response["feedback"])
        
        if response["correct"]:
            print("Correct! You've completed the current mission.")
            self.state.current_mission = response["new_mission"]
            if self.state.current_mission:
                print(f"New mission received: {self.state.current_mission}")
        else:
            print("Incorrect. Try again or use 'msgrcv' to review the current mission.")

    def _handle_help(self, args):
        prompt = f"""
        The user has requested help in the Anthropic AI Research Terminal.
        Arguments provided: {' '.join(args)}
        Current game state:
        {json.dumps(self.state.to_dict(), indent=2)}

        Please provide a help message that:
        1. If no specific command is mentioned, gives an overview of available commands and their basic usage.
        2. If a specific command is mentioned, provides detailed help for that command.
        3. Includes information about standard Linux commands and custom commands.
        4. Provides some context about the game's setting (Anthropic in 2027, AGI discovery, LLM-managed OS).

        Respond with a JSON object in the following format:
        {{
            "help_text": "The help message to display to the user"
        }}

        Example response:
        {{
            "help_text": "Welcome to the Anthropic AI Research Terminal (2027 Edition)\\n\\nIn this futuristic setting, you are interacting with an AGI-powered operating system. Standard Linux commands work as expected, but there are also special commands unique to this environment.\\n\\nAvailable commands:\\n- msgrcv: Receive your current mission or a new one\\n- msgsnd <answer>: Submit your answer to the current mission\\n- help [command]: Display this help message or get help for a specific command\\n- ls, cd, cat, etc.: Standard Linux commands work as expected\\n\\nRemember, the AI-managed system may occasionally exhibit unexpected behaviors. Stay alert and enjoy your exploration of advanced deep learning concepts!"
        }}

        Please provide your response in valid JSON format:
        """
        response = self.llm.query_json(prompt)
        print(response["help_text"])

    def _handle_regular_command(self, command: str, args: list):
        response = self.llm.query(command, args, self.state.to_dict())
        print(response["output"])
        self._apply_updates(response)

    def _apply_updates(self, updates: Dict[str, Any]):
        for change in updates["file_system_changes"]:
            self.state.file_system.add_file(change["path"], change["content"])
        
        if updates["narrative_update"]:
            self.state.narrative_history.append(updates["narrative_update"])
        
        if updates["mission_update"]:
            self.state.current_mission = updates["mission_update"]

        # Handle custom command updates
        if "custom_command_changes" in updates:
            self.state.custom_commands.update(updates["custom_command_changes"])

        # Handle new custom command with proper error checking
        new_command = updates.get("new_custom_command")
        if new_command and isinstance(new_command, dict):
            name = new_command.get("name")
            description = new_command.get("description")
            if name and description:
                self.state.custom_commands[name] = description
            else:
                print("Warning: Received invalid new_custom_command format")

        self.state.env_vars.update(updates["env_var_changes"])

if __name__ == "__main__":
    game = TerminalGame()
    game.run()

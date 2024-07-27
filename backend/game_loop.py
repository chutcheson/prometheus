# game_loop.py

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
        ]

    def handle_command(self, user_input: str) -> Dict[str, Any]:
        command, *args = user_input.split()

        if command == "exit":
            return {"terminal_output": "Thank you for using the Anthropic AI Research Terminal. Goodbye!"}
        elif command == "msgrcv":
            return self._handle_msgrcv()
        elif command == "msgsnd":
            return self._handle_msgsnd(args)
        elif command == "help":
            return self._handle_help(args)
        elif command == "clear":
            return self._handle_clear()
        else:
            return self._handle_regular_command(command, args)

    def _handle_msgrcv(self):
        response = self.llm.query_json(self._construct_msgrcv_prompt())
        
        return {
            "narrative_output": response.get('narrative', ''),
            "terminal_output": f"Current mission: {response.get('mission', 'No mission available')}",
            "current_directory": self.state.current_directory
        }

    def _handle_msgsnd(self, args):
        if not args:
            return {
                "narrative_output": "You attempt to send a message, but realize you haven't composed anything yet.",
                "terminal_output": "Usage: msgsnd <your_answer>",
                "current_directory": self.state.current_directory
            }

        answer = " ".join(args)
        response = self.llm.query_json(self._construct_msgsnd_prompt(answer))
        
        narrative = response.get('narrative', '')
        feedback = response.get('feedback', '')
        
        if response.get("correct", False):
            narrative += "\nCorrect! You've completed the current mission."
            self.state.current_mission = response.get("new_mission")
            if self.state.current_mission:
                feedback += f"\nNew mission received: {self.state.current_mission}"
        else:
            feedback += "\nIncorrect. Try again or use 'msgrcv' to review the current mission."

        return {
            "narrative_output": narrative,
            "terminal_output": feedback,
            "current_directory": self.state.current_directory
        }

    def _handle_help(self, args):
        response = self.llm.query_json(self._construct_help_prompt(args))
        
        return {
            "narrative_output": "An AI assistant materializes to provide guidance.",
            "terminal_output": response.get('help_text', 'No help available at the moment.'),
            "current_directory": self.state.current_directory
        }

    def _handle_clear(self):
        return {
            "clear_screen": True,
            "current_directory": self.state.current_directory
        }

    def _handle_regular_command(self, command: str, args: list) -> Dict[str, Any]:
        response = self.llm.query(command, args, self.state.to_dict())
        
        self._apply_updates(response)
        
        return {
            "narrative_output": response.get("narrative_output"),
            "terminal_output": response.get("terminal_output"),
            "current_directory": self.state.current_directory
        }

    def _apply_updates(self, updates: Dict[str, Any]):
        for change in updates.get("file_system_changes", []):
            self.state.file_system.add_file(change["path"], change["content"])
        
        if updates.get("narrative_update"):
            self.state.narrative_history.append(updates["narrative_update"])
        
        if updates.get("mission_update"):
            self.state.current_mission = updates["mission_update"]

        if "custom_command_changes" in updates:
            self.state.custom_commands.update(updates["custom_command_changes"])

        new_command = updates.get("new_custom_command")
        if new_command and isinstance(new_command, dict):
            name = new_command.get("name")
            description = new_command.get("description")
            if name and description:
                self.state.custom_commands[name] = description
            else:
                print("Warning: Received invalid new_custom_command format")

        self.state.env_vars.update(updates.get("env_var_changes", {}))

    def _construct_msgrcv_prompt(self):
        return f"""
       This is an educational game emulation environment. You are emulating a terminal in an Anthropic research facility in 2027 after the discovery of AGI. You are controlling the game state and creating the environment for the player. The environment should be detailed and beautiful and interesting. The main goal is to provide an advanced programmer and deep learning student (think grad student) an engaging experience to explore the dynamics at a top research lab, this includes social, financial and technical. The player should be able to interact with the environment. 

        The user has requested to receive a message (msgrcv command) in the Anthropic AI Research Terminal.
        Current game state:
        {json.dumps(self.state.to_dict(), indent=2)}

        Please provide a response that:
        1. If there's a current mission, reminds the user of the mission details.
        2. If there's no current mission, generates a new mission related to advanced deep learning concepts.
        3. Includes some narrative flavor text about the AI-managed system.

        Respond in JSON. Response format:
        {{
            "mission": "Mission details or new mission",
            "narrative": "Narrative flavor text"
        }}
        """

    def _construct_msgsnd_prompt(self, answer):
        return f"""
        This is an educational game emulation environment. You are emulating a terminal in an Anthropic research facility in 2027 after the discovery of AGI. You are controlling the game state and creating the environment for the player. The environment should be detailed and beautiful and interesting. The main goal is to provide an advanced programmer and deep learning student (think grad student) an engaging experience to explore the dynamics at a top research lab, this includes social, financial and technical. The player should be able to interact with the environment.

        The user has submitted an answer (msgsnd command) in the Anthropic AI Research Terminal.
        Current mission: {self.state.current_mission}
        User's answer: {answer}

        Please evaluate the answer and provide a response that:
        1. Determines if the answer is correct.
        2. Provides feedback on the answer.
        3. If correct, generates a new mission related to advanced deep learning concepts.
        4. Includes some narrative flavor text about the AI-managed system.

        Respond in JSON. Response format:
        {{
            "correct": true/false,
            "feedback": "Feedback on the answer",
            "new_mission": "New mission if the answer was correct, or null",
            "narrative": "Narrative flavor text"
        }}
        """

    def _construct_help_prompt(self, args):
        return f"""
        This is an educational game emulation environment. You are emulating a terminal in an Anthropic research facility in 2027 after the discovery of AGI. You are controlling the game state and creating the environment for the player. The environment should be detailed and beautiful and interesting. The main goal is to provide an advanced programmer and deep learning student (think grad student) an engaging experience to explore the dynamics at a top research lab, this includes social, financial and technical in the form of working with the terminal. This means a lot of terminal commands and research notes. The player should be able to interact with the environment.

        The user has requested help in the Anthropic AI Research Terminal.
        Arguments provided: {' '.join(args)}
        Current game state:
        {json.dumps(self.state.to_dict(), indent=2)}

        Please provide a help message that:
        1. If no specific command is mentioned, gives an overview of available commands and their basic usage.
        2. If a specific command is mentioned, provides detailed help for that command.
        3. Includes information about standard Linux commands and custom commands.
        4. Provides some context about the game's setting (Anthropic in 2027, AGI discovery, LLM-managed OS).

        Respond in JSON. Response format:
        {{
            "help_text": "The help message to display to the user"
        }}
        """

if __name__ == "__main__":
    game = TerminalGame()
    game.run()

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
        response = self.llm.query_json(self._construct_msgrcv_prompt())
        
        print(f"**Narrative response**: {response['narrative']}")
        if self.state.current_mission:
            print(f"{self.state.current_directory}: Current mission: {response['mission']}")
        else:
            self.state.current_mission = response['mission']
            print(f"{self.state.current_directory}: New mission received: {response['mission']}")

    def _handle_msgsnd(self, args):
        if not args:
            print(f"{self.state.current_directory}: stderr")
            print("Usage: msgsnd <your_answer>")
            return

        answer = " ".join(args)
        response = self.llm.query_json(self._construct_msgsnd_prompt(answer))
        
        print(f"**Narrative response**: {response['narrative']}")
        print(f"{self.state.current_directory}: {response['feedback']}")
        
        if response["correct"]:
            print("Correct! You've completed the current mission.")
            self.state.current_mission = response["new_mission"]
            if self.state.current_mission:
                print(f"New mission received: {self.state.current_mission}")
        else:
            print("Incorrect. Try again or use 'msgrcv' to review the current mission.")

    def _handle_help(self, args):
        response = self.llm.query_json(self._construct_help_prompt(args))
        print(f"**Narrative response**: An AI assistant materializes to provide guidance.")
        print(f"{self.state.current_directory}: {response['help_text']}")

    def _handle_regular_command(self, command: str, args: list):
        response = self.llm.query(command, args, self.state.to_dict())
        
        if response.get("narrative_output"):
            print(f"**Narrative response**: {response['narrative_output']}")
        
        print(f"{self.state.current_directory}: {response['terminal_output']}")
        
        self._apply_updates(response)

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

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
        # In a real implementation, load this from a JSON file
        return [
            {"question": "What is the fundamental unit of deep learning models?", "answer": "neuron"},
            {"question": "What phenomenon occurs when an LLM generates false or nonsensical information?", "answer": "hallucination"},
            # Add more missions here
        ]

    def run(self):
        print("Welcome to the Anthropic AI Research Terminal. Type 'help' for available commands.")
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
            else:
                self._handle_regular_command(command, args)

    def _handle_msgrcv(self):
        if self.state.current_mission:
            print(f"Current mission: {self.state.current_mission}")
        else:
            print("No current mission. Use 'msgsnd' to submit an answer and receive a new mission.")

    def _handle_msgsnd(self, args):
        if not args:
            print("Usage: msgsnd <your_answer>")
            return

        answer = " ".join(args)
        if not self.state.current_mission:
            self._set_new_mission()
            return

        current_mission = next((m for m in self.missions if m["question"] == self.state.current_mission), None)
        if current_mission and answer.lower() == current_mission["answer"].lower():
            print("Correct! You've completed the current mission.")
            self._set_new_mission()
        else:
            print("Incorrect. Try again or use 'msgrcv' to review the current mission.")

    def _set_new_mission(self):
        completed_missions = self.state.mission_progress.get("completed", [])
        available_missions = [m for m in self.missions if m["question"] not in completed_missions]
        
        if not available_missions:
            print("Congratulations! You've completed all available missions.")
            self.state.current_mission = None
            return

        new_mission = available_missions[0]["question"]
        self.state.current_mission = new_mission
        self.state.mission_progress.setdefault("completed", []).append(new_mission)
        print(f"New mission received: {new_mission}")

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

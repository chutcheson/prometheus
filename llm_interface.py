# llm_interface.py

import json
from typing import Dict, Any
from openai import OpenAI

class LLMInterface:
    def __init__(self):
        self.client = OpenAI()

    def query(self, command: str, args: list, game_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._construct_prompt(command, args, game_state_dict)
        response = self._call_llm(prompt)
        return self._parse_response(response)

    def _construct_prompt(self, command: str, args: list, game_state_dict: Dict[str, Any]) -> str:
        game_state_json = json.dumps(game_state_dict, indent=2)
        custom_commands = game_state_dict.get("custom_commands", {})
        
        prompt = f"""You are an AI assistant managing a terminal in a futuristic Anthropic research facility. The facility's operating system is fully controlled by an advanced language model, which can lead to occasional unexpected behaviors or inconsistencies. Your task is to provide realistic and engaging responses to user commands, maintain the game state, and progress the narrative.

Current game state:
{game_state_json}

The user has entered the command: {command} {' '.join(args)}

+Custom commands defined so far:
+{json.dumps(custom_commands, indent=2)}

Please provide a response that adheres to the following guidelines:

1. Mimic realistic terminal behavior:
   - If the command is a standard Unix/Linux command (e.g., ls, cd, cat, echo), behave as a real terminal would.
   - For standard commands, use proper formatting and typical terminal output styles.
   - If a command is improperly formatted or used incorrectly, respond with an appropriate error message similar to what a real terminal would provide.

2. Handle custom commands:
   - If the command matches a previously defined custom command, respond according to its description.
   - If it's a new custom command, create a plausible behavior for it and add it to the custom commands list.
   - Ensure consistency with previously defined custom commands.

3. Maintain narrative consistency:
   - Ensure your responses align with the existing narrative and game state.
   - Introduce subtle hints or clues related to deep learning, LLMs, or advanced AI concepts when appropriate.
   - Occasionally introduce minor glitches or unusual behaviors to reinforce the idea of an AI-controlled OS.

4. Keep responses concise but informative.

5. Update the game state as necessary:
   - Modify the file system, environment variables, or narrative elements based on the command's effects.

Respond in the following JSON format:
{{
    "output": "Text to display to the user (command output or error message)",
    "file_system_changes": [
        {{"path": "/example/file.txt", "content": "New or updated file content"}}
    ],
    "narrative_update": "New narrative event or information",
    "mission_update": "Updates to the current mission or a new mission",
    "env_var_changes": {{"VAR_NAME": "New Value"}},
    "custom_command_changes": {{"command_name": "command description"}},
    "new_custom_command": {{"name": "command_name", "description": "what the command does"}}
}}

Remember to balance realism with the unique aspects of the game world. Standard commands should behave normally, while custom commands can be more creative and tied to the game's narrative.
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def _parse_response(self, response: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(response)
            return {
                "output": parsed.get("output", ""),
                "file_system_changes": parsed.get("file_system_changes", []),
                "narrative_update": parsed.get("narrative_update", ""),
                "mission_update": parsed.get("mission_update", ""),
                "env_var_changes": parsed.get("env_var_changes", {}),
                "custom_command_changes": parsed.get("custom_command_changes", {}),
                "new_custom_command": parsed.get("new_custom_command", None)
            }
        except json.JSONDecodeError:
            print("Error: Unable to parse LLM response")  # Log parsing errors
            print("Raw response:", response)  # Log the raw response
            return {"output": "Error: Unable to parse LLM response."}

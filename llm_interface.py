# llm_interface.py

import json
from typing import Dict, Any
from openai import OpenAI

class LLMInterface:
    def __init__(self):
        self.client = OpenAI()

    def query(self, command: str, args: list, game_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._construct_prompt(command, args, game_state)
        response = self._call_llm(prompt)
        return self._parse_response(response)

    def _construct_prompt(self, command: str, args: list, game_state: Dict[str, Any]) -> str:
        game_state_json = json.dumps(game_state, indent=2)
        prompt = f"""You are an AI assistant managing a terminal in a futuristic Anthropic research facility. The facility's operating system is fully controlled by an advanced language model, which can lead to occasional unexpected behaviors or inconsistencies. Your task is to provide realistic and engaging responses to user commands, maintain the game state, and progress the narrative.

Current game state:
{game_state_json}

The user has entered the command: {command} {' '.join(args)}

Please provide:
1. The output to display to the user
2. Any updates to make to the file system
3. Any updates to the narrative or mission
4. Any changes to environment variables

Respond in the following JSON format:
{{
    "output": "Text to display to the user",
    "file_system_changes": [
        {{"path": "/example/file.txt", "content": "New or updated file content"}}
    ],
    "narrative_update": "New narrative event or information",
    "mission_update": "Updates to the current mission or a new mission",
    "env_var_changes": {{"VAR_NAME": "New Value"}}
}}

Remember:
- Maintain consistency with the existing narrative and file system state.
- Introduce subtle hints or clues related to deep learning, LLMs, or advanced AI concepts.
- Occasionally introduce minor glitches or unusual behaviors to reinforce the idea of an AI-controlled OS.
- Keep responses concise but informative.
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4",
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
                "env_var_changes": parsed.get("env_var_changes", {})
            }
        except json.JSONDecodeError:
            return {"output": "Error: Unable to parse LLM response."}

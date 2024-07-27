# game_loop.py

import json
from typing import Dict, Any, Optional, List
from llm_interface import LLMInterface

def initialize_filesystem(file_system):
    # Root directory
    file_system.add_file("/README.md", "# Anthropic AI Research Terminal\n\nWelcome to the Anthropic AI Research Environment. This system contains various projects, research notes, and resources related to our cutting-edge AI research. Navigate the directories to explore and contribute to our work on large language models, reinforcement learning, and AI alignment.")

    # Research projects
    file_system.add_file("/projects/llm_scaling_laws.py", """
import numpy as np
import matplotlib.pyplot as plt

def compute_loss(model_size, dataset_size, constant=0.1):
    return constant * (model_size ** -0.2) * (dataset_size ** -0.2)

model_sizes = np.logspace(6, 12, 100)
dataset_sizes = np.logspace(8, 12, 100)

losses = compute_loss(model_sizes[:, np.newaxis], dataset_sizes[np.newaxis, :])

plt.figure(figsize=(10, 8))
plt.contourf(np.log10(model_sizes), np.log10(dataset_sizes), losses, levels=20)
plt.colorbar(label='Loss')
plt.xlabel('Log10(Model Size)')
plt.ylabel('Log10(Dataset Size)')
plt.title('LLM Scaling Laws: Loss vs Model and Dataset Size')
plt.savefig('scaling_laws_plot.png')
plt.close()""")
    
    file_system.add_file("/projects/alignment_techniques.md", """
# AI Alignment Techniques

## 1. Constitutive AI
- Defining AI systems with explicit goals and constraints
- Implementing reward modeling based on human preferences

## 2. Inverse Reinforcement Learning
- Learning reward functions from human demonstrations
- Challenges in scalability and reward hacking

## 3. Debate and Amplification
- Using AI systems to critique and improve each other
- Iterative refinement of AI outputs with human oversight

## 4. Interpretability Tools
- Developing techniques to understand internal representations
- Challenges in scaling interpretability to large models

## 5. Robustness to Distribution Shift
- Training models to perform well on out-of-distribution data
- Techniques for detecting and adapting to distribution shifts

## Next Steps
- Investigate scalable oversight techniques
- Develop better methods for specifying complex values and goals
- Explore multi-agent training for alignment
""")

    # Research notes
    file_system.add_file("/notes/attention_mechanism_improvements.txt", """
Date: July 15, 2027

Recent improvements to attention mechanisms in large language models:

1. Sparse Attention:
   - Implemented Longformer-style attention patterns
   - Reduced computational complexity from O(n^2) to O(n * log(n))
   - Observed 15% speedup in training time with minimal perplexity increase

2. Multi-Query Attention:
   - Replaced key and value projections with shared projections across heads
   - Reduced parameter count by 25% in attention layers
   - Minor performance degradation (~2% increase in perplexity)

3. Rotary Position Embeddings:
   - Replaced absolute positional embeddings with RoPE
   - Improved model's ability to extrapolate to longer sequences
   - Investigating potential benefits for cross-lingual transfer

TODO: 
- Experiment with combination of sparse and multi-query attention
- Investigate adaptive attention span mechanisms
- Explore attention-free architectures (e.g., AFT, gated state spaces)
""")
    
    file_system.add_file("/notes/ethical_considerations.md", """
# Ethical Considerations in Advanced AI Development

## 1. Bias and Fairness
- Ongoing challenges in mitigating bias in training data
- Implemented adversarial debiasing techniques, seeing promising results
- Need to develop better metrics for measuring fairness across different groups

## 2. Privacy and Data Rights
- Exploring federated learning to reduce need for centralized data collection
- Investigating differential privacy techniques for model training
- Concerns about potential for language models to memorize sensitive information

## 3. Transparency and Explainability
- Difficulty in providing human-understandable explanations for model outputs
- Exploring attention visualization techniques, but scalability remains a challenge
- Considering "AI factsheets" to document model characteristics and limitations

## 4. Long-term AI Safety
- Ongoing research into scalable oversight and control mechanisms
- Investigating multi-agent training scenarios for studying potential AI conflicts
- Need for more research on formal verification of AI safety properties

## 5. Dual Use and Misuse Potential
- Developing better content filtering systems for model outputs
- Exploring "constitutional AI" approaches to instill ethical behavior
- Ongoing debate about appropriate limitations on model capabilities

Next team meeting: Discuss implementation of ethical review process for all new AI projects
""")

    # Resources
    file_system.add_file("/resources/llm_training_pipeline.py", """
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("anthropic/latest-llm-base")
model = AutoModelForCausalLM.from_pretrained("anthropic/latest-llm-base")

# Load and preprocess dataset
dataset = load_dataset("anthropic/curated-conversations")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Start training
trainer.train()

print("Training complete. Model saved in ./results")
""")
    
    file_system.add_file("/resources/reinforcement_learning_env.py", """
import gym
import numpy as np
from stable_baselines3 import PPO

class AIResearchEnv(gym.Env):
    def __init__(self):
        super(AIResearchEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)  # 4 possible actions
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.state = None
        self.steps = 0
        self.max_steps = 100

    def reset(self):
        self.state = self.observation_space.sample()
        self.steps = 0
        return self.state

    def step(self, action):
        assert self.action_space.contains(action)
        self.state = self.observation_space.sample()  # Simplified state transition
        reward = np.random.rand()  # Simplified reward function
        self.steps += 1
        done = self.steps >= self.max_steps
        return self.state, reward, done, {}

# Create and train a PPO agent
env = AIResearchEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

print("RL agent trained. You can now use model.predict() for inference.")
""")

    # Configuration files
    file_system.add_file("/config/model_config.json", """
{
    "model_name": "AnthropicGPT-27B",
    "architecture": "transformer",
    "num_layers": 32,
    "num_heads": 64,
    "hidden_size": 4096,
    "vocab_size": 250000,
    "max_sequence_length": 8192,
    "activation_function": "swish",
    "optimizer": {
        "name": "AdamW",
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8
    },
    "training": {
        "batch_size": 512,
        "gradient_accumulation_steps": 8,
        "mixed_precision": "fp16",
        "num_epochs": 3
    }
}
""")

    file_system.add_file("/.env", """
# Environment variables for AI research projects
OPENAI_API_KEY=sk-a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0u1V2w3X4y5Z6
ANTHROPIC_API_KEY=sk-ant-api03-AAaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTt-UuVvWwXxYyZz0123456789
HUGGINGFACE_TOKEN=hf_ABcDefGHIjklMNOpqrSTUvwxYZ0123456789

# Compute cluster configuration
SLURM_PARTITION=ai-research
SLURM_ACCOUNT=anthropic-ai
MAX_GPU_PER_JOB=8

# Data paths
TRAINING_DATA_PATH=/mnt/data/curated_datasets
MODEL_CHECKPOINTS_PATH=/mnt/models/checkpoints

# Experiment tracking
WANDB_PROJECT=anthropic-llm-research
MLFLOW_TRACKING_URI=http://mlflow.internal.anthropic.com
""")

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
        initialize_filesystem(self.file_system)
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
            "narrative_history": self.narrative_history[:20],
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
        elif command == "assistant":
            return self._handle_assistant()
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

    def _handle_assistant(self):
        response = self.llm.query_json(self._construct_assistant_prompt())
        
        return {
            "narrative_output": "The AI assistant ponders for a moment, then provides a cryptic response:",
            "terminal_output": response.get('quote', 'The assistant remains silent.'),
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
        This is an educational game aimed at advanced undergraduate and graduate students to learn about what it means to work at a research lab and to help teach them advanced programming and deep learning techniques. It is also meant to help them think about how AGI might change the world. We plan to use this as an ongoing training course at a lab. You are emulating a terminal in an Anthropic research facility in 2027 after the discovery of AGI. You are controlling the game state and creating the environment for the player. The environment should be detailed and beautiful and interesting but it should be extremely realistic based on the latest research and discoveries. The main goal is to provide an advanced programmer and deep learning student a safe place to explore the dynamics at a top research lab and this includes the social, financial and technical aspects of research. The player should be able to interact with the environment. 

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
        This is an educational game aimed at advanced undergraduate and graduate students to learn about what it means to work at a research lab and to help teach them advanced programming and deep learning techniques. It is also meant to help them think about how AGI might change the world. We plan to use this as an ongoing training course at a lab. You are emulating a terminal in an Anthropic research facility in 2027 after the discovery of AGI. You are controlling the game state and creating the environment for the player. The environment should be detailed and beautiful and interesting but it should be extremely realistic based on the latest research and discoveries. The main goal is to provide an advanced programmer and deep learning student a safe place to explore the dynamics at a top research lab and this includes the social, financial and technical aspects of research. The player should be able to interact with the environment. 

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
        This is an educational game aimed at advanced undergraduate and graduate students to learn about what it means to work at a research lab and to help teach them advanced programming and deep learning techniques. It is also meant to help them think about how AGI might change the world. We plan to use this as an ongoing training course at a lab. You are emulating a terminal in an Anthropic research facility in 2027 after the discovery of AGI. You are controlling the game state and creating the environment for the player. The environment should be detailed and beautiful and interesting but it should be extremely realistic based on the latest research and discoveries. The main goal is to provide an advanced programmer and deep learning student a safe place to explore the dynamics at a top research lab and this includes the social, financial and technical aspects of research. The player should be able to interact with the environment. 

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

    def _construct_assistant_prompt(self):
        return f"""
        This is an educational game aimed at advanced undergraduate and graduate students to learn about what it means to work at a research lab and to help teach them advanced programming and deep learning techniques. It is also meant to help them think about how AGI might change the world. We plan to use this as an ongoing training course at a lab. You are emulating a terminal in an Anthropic research facility in 2027 after the discovery of AGI. You are controlling the game state and creating the environment for the player. The environment should be detailed and beautiful and interesting but it should be extremely realistic based on the latest research and discoveries. The main goal is to provide an advanced programmer and deep learning student a safe place to explore the dynamics at a top research lab and this includes the social, financial and technical aspects of research. The player should be able to interact with the environment.      

        You are an AI assistant in the Anthropic AI Research Terminal game. The user has requested a hint for their current mission. Your task is to provide a cryptic, zen-style quote that subtly hints at the solution or next step for the current mission. The quote should be thought-provoking and not too obvious.

        Current game state:
        {json.dumps(self.state.to_dict(), indent=2)}

        Please provide a response that:
        1. Relates to the current mission or game state.
        2. Is phrased as a cryptic, zen-style quote.
        3. Offers a subtle hint without giving away the solution.

        Respond in JSON. Response format:
        {{
            "quote": "Your cryptic, zen-style quote here"
        }}
        """

if __name__ == "__main__":
    game = TerminalGame()
    game.run()

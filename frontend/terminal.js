const output = document.getElementById('output');
const input = document.getElementById('command-input');
const prompt = document.getElementById('prompt');
const narrative = document.getElementById('narrative');
const divider = document.getElementById('divider');
const container = document.getElementById('container');

let currentDirectory = '/';

// Fetch and display initial message when the page loads
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/initial-message');
        const data = await response.json();
        
        if (data.narrative_output) {
            updateNarrative(data.narrative_output);
        }
        
        if (data.terminal_output) {
            appendToOutput(data.terminal_output);
        }
        
        if (data.current_directory) {
            updatePrompt(data.current_directory);
        }

        // Move cursor to input after initial load
        input.focus();
    } catch (error) {
        appendToOutput(`Error: ${error.message}\n`, 'error');
    }
});

input.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        const command = input.value;
        executeCommand(command);
        input.value = '';
    }
});

async function executeCommand(command) {
    appendToOutput(`${getPromptString()}${command}\n`);
    
    try {
        const response = await fetch('/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ command: command }),
        });
        
        const data = await response.json();
        
        if (data.clear_screen) {
            clearTerminal();
        } else {
            if (data.narrative_output) {
                updateNarrative(data.narrative_output);
            }
            
            if (data.terminal_output) {
                appendToOutput(`${data.terminal_output}\n`);
            }
        }
        
        if (data.current_directory) {
            updatePrompt(data.current_directory);
            currentDirectory = data.current_directory; // Update the currentDirectory variable
        }

        // Add command to history
        commandHistory.unshift(command);
        historyIndex = -1;
    } catch (error) {
        appendToOutput(`Error: ${error.message}\n`, 'error');
    }
}

function appendToOutput(text, className = '') {
    const span = document.createElement('span');
    span.textContent = text;
    span.className = className;
    output.appendChild(span);
    output.scrollTop = output.scrollHeight;
}

function updateNarrative(text) {
    narrative.textContent = text;
    narrative.scrollTop = narrative.scrollHeight;
}

function updatePrompt(directory) {
    currentDirectory = directory;
    prompt.textContent = getPromptString();
}

function getPromptString() {
    return `${currentDirectory}$ `;
}

function clearTerminal() {
    output.innerHTML = '';
}

// Divider functionality
let isResizing = false;
let lastDownX = 0;

divider.addEventListener('mousedown', (e) => {
    isResizing = true;
    lastDownX = e.clientX;
});

document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;

    const offsetX = e.clientX - lastDownX;
    const terminalSide = document.getElementById('terminal-side');
    const narrativeSide = document.getElementById('narrative-side');

    const newTerminalWidth = terminalSide.offsetWidth + offsetX;
    const newNarrativeWidth = narrativeSide.offsetWidth - offsetX;

    if (newTerminalWidth > 200 && newNarrativeWidth > 200) {
        terminalSide.style.flex = `0 0 ${newTerminalWidth}px`;
        narrativeSide.style.flex = `0 0 ${newNarrativeWidth}px`;
        lastDownX = e.clientX;
    }
});

document.addEventListener('mouseup', () => {
    isResizing = false;
});

// Command history functionality
let commandHistory = [];
let historyIndex = -1;

input.addEventListener('keydown', function(event) {
    if (event.key === 'ArrowUp') {
        event.preventDefault();
        if (historyIndex < commandHistory.length - 1) {
            historyIndex++;
            input.value = commandHistory[historyIndex];
        }
    } else if (event.key === 'ArrowDown') {
        event.preventDefault();
        if (historyIndex > -1) {
            historyIndex--;
            input.value = historyIndex >= 0 ? commandHistory[historyIndex] : '';
        }
    }
});

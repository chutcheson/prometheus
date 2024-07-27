const output = document.getElementById('output');
const input = document.getElementById('command-input');
const prompt = document.getElementById('prompt');

input.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        const command = input.value;
        executeCommand(command);
        input.value = '';
    }
});

async function executeCommand(command) {
    appendToOutput(`$ ${command}\n`);
    
    try {
        const response = await fetch('/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ command: command }),
        });
        
        const data = await response.json();
        
        if (data.narrative_output) {
            appendToOutput(`${data.narrative_output}\n`, 'narrative');
        }
        
        if (data.terminal_output) {
            appendToOutput(`${data.terminal_output}\n`);
        }
        
        updatePrompt(data.current_directory);
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

function updatePrompt(directory) {
    prompt.textContent = `${directory}$`;
}

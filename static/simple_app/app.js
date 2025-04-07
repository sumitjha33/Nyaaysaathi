async function login() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        if (data.message) {
            document.getElementById('login-area').style.display = 'none';
            document.getElementById('chat-area').style.display = 'block';
            loadHistory();
        } else {
            alert(data.error || 'Login failed');
        }
    } catch (e) {
        alert('Login failed');
    }
}

async function sendMessage() {
    const messageInput = document.getElementById('message');
    const message = messageInput.value;
    if (!message) return;

    addMessage(message, true);
    messageInput.value = '';

    try {
        const formData = new FormData();
        formData.append('message', message);

        const response = await fetch('/chat', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        addMessage(data.response, false);
    } catch (e) {
        addMessage('Error: Could not get response', false);
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/history');
        const history = await response.json();
        
        history.forEach(conv => {
            addMessage(conv.message, true);
            addMessage(conv.response, false);
        });
    } catch (e) {
        console.error('Could not load history');
    }
}

function addMessage(text, isUser) {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = `message ${isUser ? 'user' : 'bot'}`;
    div.textContent = text;
    messages.appendChild(div);
    div.scrollIntoView({ behavior: 'smooth' });
}

// Handle Enter key in message input
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('message').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
});

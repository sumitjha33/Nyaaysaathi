function addMessage(message, isUser) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.innerHTML = message; // Changed from textContent to innerHTML
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    
    if (message) {
        addMessage(message, true);
        input.value = '';
        
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `message=${encodeURIComponent(message)}`
        })
        .then(response => response.json())
        .then(data => {
            // Format and display the bot's response
            const formattedResponse = data.response.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            addMessage(formattedResponse, false);
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage('Sorry, something went wrong. Please try again.', false);
        });
    }
}

document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Add login handling
function handleLogin(e) {
    e.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    fetch('/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            document.getElementById('login-container').style.display = 'none';
            document.getElementById('chat-container').style.display = 'block';
            loadUserHistory();
        }
    });
}

function loadUserHistory() {
    fetch('/history')
        .then(response => response.json())
        .then(history => {
            history.forEach(conv => {
                addMessage(conv.message, true);
                addMessage(conv.response, false);
            });
        });
}

// Add event listener for login form
document.getElementById('login-form').addEventListener('submit', handleLogin);

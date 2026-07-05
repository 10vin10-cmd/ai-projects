document.addEventListener('DOMContentLoaded', () => {
    // DOM Element Targets
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const modelSelect = document.getElementById('modelSelect');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const messagesContainer = document.getElementById('messagesContainer');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const messagesEnd = document.getElementById('messagesEnd');
    const clearBtn = document.getElementById('clearBtn');
    const sendIcon = document.getElementById('sendIcon');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const sendButton = document.getElementById('sendButton');

    // Auto-adjust text input boundaries gracefully
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'x';
        if (this.value === '') {
            this.style.height = 'auto';
        }
    });

    // Capture explicit keystrokes (Enter to submit message vs Shift+Enter for new line)
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.requestSubmit();
        }
    });

    // Handle incoming app workflow tracking configurations
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const messageText = messageInput.value.trim();
        if (!messageText) return;

        const selectedModel = modelSelect.value;

        // Hide landing instructions and commit message rendering loops
        welcomeScreen.style.display = 'none';
        clearBtn.style.display = 'flex';
        appendMessage(messageText, 'user');

        // Reset text targets immediately
        messageInput.value = '';
        messageInput.style.height = 'auto';

        // Toggle state animations to active processing
        setLoadingState(true);
        scrollToBottom();

        try {
            // Network Request Payload execution matching Flask specs
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: messageText,
                    model: selectedModel
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP network runtime fault: ${response.status}`);
            }

            const data = await response.json();
            
            // Render engine handles structured JSON blocks or fallback strings cleanly
            if (data && data.response) {
                appendMessage(data.response, 'ai', data.summary, data.sentiment);
            } else {
                appendMessage("Received an empty or misconfigured server layout.", 'ai');
            }

        } catch (error) {
            console.error("Payload route error details:", error);
            appendMessage(`Error: Could not retrieve response (${error.message})`, 'ai');
        } finally {
            setLoadingState(false);
            scrollToBottom();
        }
    });

    // Event Handler: Clear Session Conversions
    clearBtn.addEventListener('click', () => {
        messagesContainer.innerHTML = '';
        welcomeScreen.style.display = 'block';
        clearBtn.style.display = 'none';
    });

    // Helper Utility: Dom Layout Injections
    function appendMessage(text, sender, summary = null, sentiment = null) {
        const wrapper = document.createElement('div');
        wrapper.classList.add('message-wrapper', sender);

        // If your AI backend generates structured items (summary/sentiment) render metrics
        if (sender === 'ai' && (summary || sentiment !== null)) {
            let metaHeader = `<div class="structured-meta">`;
            if (summary) metaHeader += `<strong>Summary:</strong> ${summary} | `;
            if (sentiment !== null) metaHeader += `<strong>Sentiment:</strong> ${sentiment}/100`;
            metaHeader += `</div>`;
            wrapper.innerHTML = metaHeader + `<div class="message-text">${text}</div>`;
        } else {
            wrapper.textContent = text;
        }

        messagesContainer.appendChild(wrapper);
    }

    // Helper Utility: Loading Visual Flags
    function setLoadingState(isLoading) {
        if (isLoading) {
            loadingIndicator.style.display = 'block';
            sendIcon.style.display = 'none';
            loadingSpinner.style.display = 'block';
            sendButton.disabled = true;
        } else {
            loadingIndicator.style.display = 'none';
            sendIcon.style.display = 'block';
            loadingSpinner.style.display = 'none';
            sendButton.disabled = false;
            messageInput.focus();
        }
    }

    function scrollToBottom() {
        messagesEnd.scrollIntoView({ behavior: 'smooth' });
    }
});

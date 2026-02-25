// LLM Chat Interface Management

const LLMInterface = {
    messageHistory: [],

    /**
     * Initialize chat interface
     */
    init() {
        const sendButton = document.getElementById('send-message');
        const chatInput = document.getElementById('chat-input');

        if (sendButton) {
            sendButton.addEventListener('click', () => this.sendMessage());
        }

        if (chatInput) {
            chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        console.log('LLM Interface initialized');
    },

    /**
     * Send user message
     */
    sendMessage() {
        const chatInput = document.getElementById('chat-input');
        if (!chatInput) return;

        const message = chatInput.value.trim();
        if (!message) return;

        // Add user message to history
        this.addMessage('user', message);

        // Clear input
        chatInput.value = '';

        // Generate AI response
        setTimeout(() => {
            const response = LLMEngine.generateResponse(message);
            this.addMessage('assistant', response);

            // Check if response includes protocol application
            if (response.includes('apply these settings')) {
                this.showApplyButton();
            }
        }, 500);
    },

    /**
     * Add message to chat
     */
    addMessage(role, content) {
        const messagesContainer = document.getElementById('chat-messages');
        if (!messagesContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        // Process markdown-like formatting
        let formattedContent = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');

        contentDiv.innerHTML = formattedContent;
        messageDiv.appendChild(contentDiv);

        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Store in history
        this.messageHistory.push({ role, content });
    },

    /**
     * Show apply settings button
     */
    showApplyButton() {
        const messagesContainer = document.getElementById('chat-messages');
        if (!messagesContainer) return;

        const buttonDiv = document.createElement('div');
        buttonDiv.className = 'message assistant';
        buttonDiv.innerHTML = `
            <button class="btn btn-primary" onclick="LLMInterface.applyRecommendedSettings()">
                ✓ Apply Recommended Settings
            </button>
        `;

        messagesContainer.appendChild(buttonDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    },

    /**
     * Apply AI-recommended settings
     */
    applyRecommendedSettings() {
        // Get last protocol recommendation
        const lastUserMessage = this.messageHistory
            .filter(m => m.role === 'user')
            .slice(-1)[0];

        if (!lastUserMessage) return;

        const protocol = LLMEngine.generateProtocol(lastUserMessage.content);

        // Apply parallel imaging settings
        if (window.ParallelImaging && protocol.parameters) {
            ParallelImaging.applyConfig(protocol.parameters);
        }

        // Provide feedback
        this.addMessage('assistant', '✓ Settings applied! You can now review the updated parameters above.');
    },

    /**
     * Clear chat history
     */
    clearChat() {
        const messagesContainer = document.getElementById('chat-messages');
        if (messagesContainer) {
            messagesContainer.innerHTML = '';
        }
        this.messageHistory = [];
    }
};

// Export
window.LLMInterface = LLMInterface;

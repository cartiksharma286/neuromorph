const questionInput = document.getElementById('question');
const form = document.getElementById('chat-form');
const chatWindow = document.getElementById('chat-window');
const statusLabel = document.getElementById('status-label');

function addMessage(role, text) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'YOU' : 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  if (role === 'user') {
    wrapper.appendChild(bubble);
    wrapper.appendChild(avatar);
  } else {
    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
  }

  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function setStatus(text) {
  statusLabel.textContent = text;
}

async function sendQuestion(event) {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage('user', question);
  questionInput.value = '';
  questionInput.style.height = 'auto';
  setStatus('Thinking...');

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });

    const data = await response.json();
    const answer = data.answer || 'No answer returned.';
    addMessage('bot', answer);
    setStatus('Ready');
  } catch (error) {
    addMessage('bot', 'Something went wrong while contacting the assistant. Please try again.');
    setStatus('Error');
  }
}

form.addEventListener('submit', sendQuestion);
questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

questionInput.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 170) + 'px';
});

const chips = document.querySelectorAll('.chip');
chips.forEach((chip) => {
  chip.addEventListener('click', () => {
    questionInput.value = chip.textContent;
    questionInput.focus();
  });
});

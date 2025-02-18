import './style.css';
import { sendMessage, adjustInputWidth } from './send.ts';

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div>
    <!-- Checkbox -->
    <div class="history-chat-output">Welcome to Pizza Topping Basic demonstration. You can order a pizza with selected sizes, types, and toppings. Ask for help if needed.</div>
    <!-- chat history -->
    <div id="chat-history" class="chat-history"></div>
    <!-- input chat -->
    <div class="input-group">
      <span id="measure-span" class="measure-span"></span>
      <input type="text" id="chat-input" class="chat-input" placeholder="Type your message here" />
      <button id="send" class="btn btn-primary">Send</button>
    </div>
    <div class="form-check">
      <label class="form-check-label ms-2" for="use-gemini">
      Use Gemini
      </label>
      <input class="form-check-input" type="checkbox" value="true" id="use-gemini" checked>
    </div>
    <div id="chat-container" class="chat-history"></div>
  </div>
`;

document.addEventListener('DOMContentLoaded', () => {
  // Select elements
  const chatInput = document.querySelector<HTMLInputElement>('#chat-input')!;
  const measureSpan = document.querySelector<HTMLSpanElement>('#measure-span')!;
  const sendButton = document.querySelector<HTMLButtonElement>('#send')!;
  const chatHistoryElement = document.querySelector<HTMLDivElement>('#chat-history')!;
  const chatContainer = document.querySelector<HTMLDivElement>('#chat-container')!;
  const toggleElement = document.querySelector<HTMLInputElement>('#use-gemini')!;

  // Adjust input width on load
  adjustInputWidth(chatInput, measureSpan);

  // Attach width adjustment to input
  chatInput.addEventListener('input', () => adjustInputWidth(chatInput, measureSpan));

  // Handle Enter key press for sending
  chatInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
      sendButton.click(); // Simulate button click to trigger `sendMessage`
    }
  });

  // Set up message sending with the selected elements
  sendMessage(sendButton, chatInput, chatHistoryElement, measureSpan, chatContainer, toggleElement);
});

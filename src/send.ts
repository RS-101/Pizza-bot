export function sendMessage(
  element: HTMLButtonElement,
  inputElement: HTMLInputElement,
  chatHistoryElement: HTMLDivElement,
  measureSpan: HTMLSpanElement,
  chatContainer: HTMLDivElement,
  toggleElement: HTMLInputElement // Checkbox for toggling use_gemini
) {
  element.addEventListener("click", async () => {
    const message = inputElement.value.trim();
    if (!message) return; // Don't send empty messages

    // Create new chat entry container but don't append it yet
    const chatEntry = document.createElement("div");
    chatEntry.classList.add("chat-entry");

    const useGemini = toggleElement.checked; // Get the toggle state

    // Create loading indicator
    const loadingIndicator = document.createElement("div");
    loadingIndicator.classList.add("history-chat-output");
    loadingIndicator.textContent = "I am thinking ";
    const frames = ["|", "/", "-", "\\"];
    let frameIndex = 0;

    // Start loading animation
    const interval = setInterval(() => {
      loadingIndicator.textContent = `I am thinking ${frames[frameIndex]}`;
      frameIndex = (frameIndex + 1) % frames.length;
    }, 200);

    chatContainer.appendChild(loadingIndicator);

    try {
      // Wait for backend response
      const response = await fetchData(message, useGemini);

      // Stop loading animation
      clearInterval(interval);
      chatContainer.removeChild(loadingIndicator);

      // Add user message and response only after thinking is done
      chatEntry.innerHTML = `<div class="history-chat-input">${message}</div>`;
      const responseElement = document.createElement("div");
      responseElement.classList.add("history-chat-output");
      const formattedResponse = formatResponseText(response);
      responseElement.innerHTML = formattedResponse;
      console.log("message fetched and formatted:", formattedResponse);

      chatEntry.appendChild(responseElement);

      chatHistoryElement.appendChild(chatEntry);
    } catch (error) {
      console.error("Error fetching message:", error);
    }

    // Clear input field
    inputElement.value = "";

    // Reset input width
    adjustInputWidth(inputElement, measureSpan);

    // Auto-scroll to latest chat entry
    chatHistoryElement.scrollTop = chatHistoryElement.scrollHeight;
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "instant" });
  });
}



// Adjust input width dynamically
export function adjustInputWidth(input: HTMLInputElement, measureSpan: HTMLSpanElement) {
  measureSpan.textContent = input.value || input.placeholder;
  measureSpan.style.font = window.getComputedStyle(input).font;
  input.style.width = `${measureSpan.offsetWidth-15}px`; // Adjust for padding
}

import axios from "axios";

const API_URL = "http://127.0.0.1:8000"; // Python Backend URL

export const fetchData = async (user_message: string, useGemini: boolean) => {  try {
    const response = await axios.post(`${API_URL}/api/data`, {
      message: user_message,
      use_gemini: useGemini,
    });
    console.log("Response fetched:", response);
    console.log("Data fetched:", response.data);
    return response.data.message;
  } catch (error) {
    console.error("Error fetching data:", error);
    throw error;
  }
};


function formatResponseText(response: string): string {
  // Convert <think> block to a div with class
  response = response.replace(/<think>([\s\S]*?)<\/think>/g, (_, content) => {
      const formattedContent = content.trim().replace(/\n{2,}/g, '<br><br>'); // Preserve paragraph breaks
      return `<div class="think">${formattedContent}</div>`;
  });

  // Convert **bold text** to <strong>text</strong>
  response = response.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

  // Convert numbered list into HTML <ol> structure while preserving line breaks
  response = response.replace(/\d+\.\s\*\*(.*?)\*\*:\s(.*?)\./g, "<li><strong>$1</strong>: $2.</li>");

  // Ensure each sentence stays on a new line by converting `\n` to `<br>`
  response = response.replace(/\n/g, "<br>");

  // Wrap list items in <ol> if they exist
  if (response.includes("<li>")) {
      response = response.replace(/(<li>.*?<\/li>)/gs, "<ol>$1</ol>");
  }

  return response;
}

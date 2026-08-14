const conversation = document.getElementById("conversation");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const newChatButton = document.getElementById("newChatButton");
const connectionDot = document.getElementById("connectionDot");
const connectionLabel = document.getElementById("connectionLabel");
const welcomeTemplate = document.getElementById("welcomeTemplate");

let isSending = false;

function resetConversation() {
  conversation.replaceChildren(welcomeTemplate.content.cloneNode(true));
  messageInput.value = "";
  resizeInput();
  messageInput.focus();
}

function setConnection(online) {
  connectionDot.classList.toggle("is-ready", online);
  connectionDot.classList.toggle("is-offline", !online);
  connectionDot.classList.remove("is-checking");
  connectionLabel.textContent = online ? "Ready" : "Offline";
}

function appendMessage(kind, text, label = "Assistant") {
  const row = document.createElement("div");
  row.className = `message-row ${kind}`;

  const message = document.createElement("div");
  message.className = "message";

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = label;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  message.append(meta, bubble);
  row.append(message);
  conversation.append(row);
  row.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return row;
}

function appendTyping() {
  const row = document.createElement("div");
  row.className = "message-row assistant typing";
  row.innerHTML = `
    <div class="message">
      <div class="message-meta">Assistant</div>
      <div class="bubble" aria-label="Assistant is responding">
        <span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span>
      </div>
    </div>`;
  conversation.append(row);
  row.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return row;
}

function resizeInput() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 140)}px`;
}

async function checkHealth() {
  try {
    const response = await fetch("/health", { cache: "no-store" });
    setConnection(response.ok);
  } catch (error) {
    setConnection(false);
  }
}

async function sendMessage(rawMessage) {
  const message = rawMessage.trim();
  if (!message || isSending) return;

  isSending = true;
  sendButton.disabled = true;
  messageInput.disabled = true;
  appendMessage("user", message, "You");
  messageInput.value = "";
  resizeInput();
  const typingRow = appendTyping();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    typingRow.remove();

    if (!response.ok) {
      appendMessage("notice", data.detail || "The request could not be processed.", "Notice");
      return;
    }

    appendMessage("assistant", data.response || "The assistant returned no response.");
    setConnection(true);
  } catch (error) {
    typingRow.remove();
    appendMessage(
      "notice",
      "The local assistant is unavailable. Check that Ollama and CampusBot are running.",
      "Notice",
    );
    setConnection(false);
  } finally {
    isSending = false;
    sendButton.disabled = false;
    messageInput.disabled = false;
    messageInput.focus();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(messageInput.value);
});

messageInput.addEventListener("input", resizeInput);
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

conversation.addEventListener("click", (event) => {
  const suggestion = event.target.closest("[data-message]");
  if (suggestion) sendMessage(suggestion.dataset.message);
});

newChatButton.addEventListener("click", resetConversation);

resetConversation();
checkHealth();


import { useEffect, useRef, useState } from "react";


const API_URL = "/api/chat";


function newSessionId() {
  if (globalThis.crypto?.randomUUID) {
    return `s_${globalThis.crypto.randomUUID().replaceAll("-", "")}`;
  }
  return `s_${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`;
}


export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "agent",
      text: "Tell me the cuisine, high/low tier or budget in yuan, date, time, and party size.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const sessionId = useRef(newSessionId());
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, error]);

  async function submit(event) {
    event.preventDefault();
    const message = input.trim();
    if (!message || loading) return;

    setMessages((current) => [...current, { role: "user", text: message }]);
    setInput("");
    setError("");
    setLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId.current, message }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || `The server returned ${response.status}.`);
      }
      setMessages((current) => [
        ...current,
        { role: "agent", text: data.reply, confirmed: data.done },
      ]);
    } catch (requestError) {
      setError(
        requestError.message === "Failed to fetch"
          ? "Cannot reach the reservation service. Check that the backend is running on port 8000."
          : requestError.message,
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <section className="chat" aria-label="Restaurant reservation chat">
        <header>
          <h1>Restaurant Reservations</h1>
          <p>Chat with the booking agent</p>
        </header>

        <div className="messages" aria-live="polite">
          {messages.map((message, index) => (
            <div key={`${message.role}-${index}`} className={`message ${message.role}`}>
              <span>{message.role === "user" ? "You" : "Agent"}</span>
              <p>{message.text}</p>
              {message.confirmed && <strong>Reservation made</strong>}
            </div>
          ))}
          {loading && <p className="loading">Checking your request…</p>}
          {error && <p className="error">{error}</p>}
          <div ref={endRef} />
        </div>

        <form onSubmit={submit}>
          <label htmlFor="message" className="sr-only">
            Reservation request
          </label>
          <textarea
            id="message"
            rows="2"
            value={input}
            disabled={loading}
            placeholder="Example: Korean for 4 tomorrow at 19:00, budget 80 yuan"
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            {loading ? "Sending…" : "Submit"}
          </button>
        </form>
      </section>
    </main>
  );
}

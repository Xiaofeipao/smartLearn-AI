import { useState } from "react";
import { askQuestion } from "./api";

export default function ChatPanel({ chatId, enabled, onBusy, onJumpToPage }) {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleAsk() {
    const q = message.trim();
    if (!q || loading) return;
    setLoading(true);
    setError(null);
    if (onBusy) onBusy(true);

    const userMsg = { role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setMessage("");

    try {
      const data = await askQuestion(q, chatId);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          citations: data.citations || [],
          sources: data.sources || [],
        },
      ]);
    } catch (e) {
      setError(e.message);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${e.message}`, citations: [], sources: [] },
      ]);
    } finally {
      setLoading(false);
      if (onBusy) onBusy(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="chat-empty">Ask a question about the uploaded PDF.</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`msg msg-${msg.role}`}>
            <div className="msg-role">{msg.role === "user" ? "You" : "AI"}</div>
            <p className="msg-content">{msg.content}</p>
            {msg.citations && msg.citations.length > 0 && (
              <div className="msg-citations">
                {msg.citations.map((page) => (
                  <button
                    key={page}
                    className="citation-link"
                    onClick={() => onJumpToPage && onJumpToPage(page)}
                  >
                    Page {page}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {error && <div className="error-banner" role="alert">{error}</div>}
      {loading && <p className="status-text">Asking…</p>}

      <div className="chat-input-row">
        <textarea
          rows={2}
          value={message}
          placeholder="Type your question…"
          disabled={!enabled || loading}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          type="button"
          onClick={handleAsk}
          disabled={!enabled || !message.trim() || loading}
        >
          Ask
        </button>
      </div>
    </div>
  );
}

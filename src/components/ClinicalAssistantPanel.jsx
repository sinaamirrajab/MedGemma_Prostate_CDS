import { useState } from "react";

export default function ClinicalAssistantPanel({
  patient,
  modelPrediction,
  recommendation,
}) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [isSending, setIsSending] = useState(false);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isSending) {
      return;
    }

    const nextMessages = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);
    setInput("");
    setIsSending(true);

    try {
      const response = await fetch("/api/medgemma/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          messages: nextMessages,
          patient,
          modelPrediction,
          recommendation: recommendation || {},
        }),
      });
      if (!response.ok) {
        throw new Error(`Chat request failed: ${response.status}`);
      }
      const data = await response.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Unable to reach MedGemma chat right now. Please retry.",
        },
      ]);
      console.error(error);
    } finally {
      setIsSending(false);
    }
  };

  const onKeyDown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <section className="card chatbot">
      <div className="card-header">
        <h2>Clinical Assistant</h2>
        <p>Ask MedGemma about the case, guidelines, or next steps.</p>
      </div>
      <div className="chatbot-shell">
        <div className="chatbot-messages">
          {messages.length === 0 ? (
            <div className="chatbot-empty">Chat history will appear here.</div>
          ) : (
            messages.map((message, index) => (
              <div key={`${message.role}-${index}`} className={`chat-row ${message.role}`}>
                <span className="chat-role">{message.role === "user" ? "You" : "MedGemma"}</span>
                <p>{message.content}</p>
              </div>
            ))
          )}
        </div>
        <div className="chatbot-input">
          <input
            type="text"
            placeholder="Ask about treatment options, evidence, or imaging findings..."
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={onKeyDown}
          />
          <button type="button" onClick={sendMessage} disabled={isSending}>
            {isSending ? "Sending..." : "Send"}
          </button>
        </div>
      </div>
    </section>
  );
}

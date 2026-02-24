import { useState } from "react";

/** Minimal markdown → HTML for chat bubbles.
 *  Handles: **bold**, *italic*, `code`, bullet lists (* / -), numbered lists, newlines.
 *  No external dependency needed. */
function markdownToHtml(text) {
  if (!text) return "";
  let html = text
    // Escape raw HTML to prevent injection
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    // Bold **text** or __text__
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.+?)__/g, "<strong>$1</strong>")
    // Italic *text* or _text_  (after bold so ** is consumed first)
    .replace(/\*([^*\n]+?)\*/g, "<em>$1</em>")
    .replace(/_([^_\n]+?)_/g, "<em>$1</em>")
    // Inline code `code`
    .replace(/`([^`]+)`/g, "<code>$1</code>");

  // Convert lines to structured HTML
  const lines = html.split("\n");
  const out = [];
  let inList = false;
  let listType = null;

  for (const raw of lines) {
    const bulletMatch = raw.match(/^\s*[*-]\s+(.*)/);
    const numberedMatch = raw.match(/^\s*\d+\.\s+(.*)/);
    if (bulletMatch) {
      if (!inList || listType !== "ul") {
        if (inList) out.push(`</${listType}>`);
        out.push("<ul>");
        inList = true;
        listType = "ul";
      }
      out.push(`<li>${bulletMatch[1]}</li>`);
    } else if (numberedMatch) {
      if (!inList || listType !== "ol") {
        if (inList) out.push(`</${listType}>`);
        out.push("<ol>");
        inList = true;
        listType = "ol";
      }
      out.push(`<li>${numberedMatch[1]}</li>`);
    } else {
      if (inList) {
        out.push(`</${listType}>`);
        inList = false;
        listType = null;
      }
      out.push(raw === "" ? "<br/>" : `<p>${raw}</p>`);
    }
  }
  if (inList) out.push(`</${listType}>`);
  return out.join("");
}

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
                {message.role === "assistant" ? (
                  <div
                    className="chat-markdown"
                    dangerouslySetInnerHTML={{ __html: markdownToHtml(message.content) }}
                  />
                ) : (
                  <p>{message.content}</p>
                )}
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

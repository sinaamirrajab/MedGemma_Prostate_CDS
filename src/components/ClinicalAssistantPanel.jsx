export default function ClinicalAssistantPanel() {
  return (
    <section className="card chatbot">
      <div className="card-header">
        <h2>Clinical Assistant</h2>
        <p>Ask MedGemma about the case, guidelines, or next steps.</p>
      </div>
      <div className="chatbot-shell">
        <div className="chatbot-messages">
          <div className="chatbot-empty">Chat history will appear here.</div>
        </div>
        <div className="chatbot-input">
          <input
            type="text"
            placeholder="Ask about treatment options, evidence, or imaging findings..."
          />
          <button type="button">Send</button>
        </div>
      </div>
    </section>
  );
}

export default function TopBar() {
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark">MG</span>
        <div>
          <p className="brand-title">MedGemma Impact Challenge</p>
          <p className="brand-subtitle">Clinical Decision Support Prototype</p>
        </div>
      </div>
      <button className="ghost-btn" type="button">
        Export Report
      </button>
    </header>
  );
}

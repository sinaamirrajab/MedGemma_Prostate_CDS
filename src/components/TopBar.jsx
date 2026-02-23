export default function TopBar({ onExportReport, isReportDisabled = false }) {
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark">PM</span>
        <div>
          <p className="brand-title">The MedGemma Impact Challenge 2026</p>
          <p className="brand-subtitle">Precision Medicine Maastricht University</p>
        </div>
      </div>
      <div className="topbar-actions">
        <button
          className="ghost-btn"
          type="button"
          onClick={onExportReport}
          disabled={isReportDisabled}
        >
          Generate PDF report
        </button>
      </div>
    </header>
  );
}

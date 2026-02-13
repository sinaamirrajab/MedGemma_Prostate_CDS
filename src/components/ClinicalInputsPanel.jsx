export default function ClinicalInputsPanel({ form, onChange, onSubmit }) {
  return (
    <section className="card form">
      <div className="card-header">
        <h2>Clinical Inputs</h2>
        <p>Capture TNM staging and key variables.</p>
      </div>
      <form onSubmit={onSubmit} className="input-grid">
        <label>
          cT stage
          <select name="tStage" value={form.tStage} onChange={onChange}>
            <option>T1</option>
            <option>T2</option>
            <option>T3</option>
            <option>T4</option>
          </select>
        </label>
        <label>
          cN stage
          <select name="nStage" value={form.nStage} onChange={onChange}>
            <option>N0</option>
            <option>N1</option>
          </select>
        </label>
        <label>
          cM stage
          <select name="mStage" value={form.mStage} onChange={onChange}>
            <option>M0</option>
            <option>M1</option>
          </select>
        </label>
        <label>
          PSA (ng/mL)
          <input name="psa" value={form.psa} onChange={onChange} />
        </label>
        <label>
          Performance status
          <select name="performance" value={form.performance} onChange={onChange}>
            <option>ECOG 0</option>
            <option>ECOG 1</option>
            <option>ECOG 2</option>
            <option>ECOG 3</option>
          </select>
        </label>
        <label>
          Age
          <input name="age" value={form.age} onChange={onChange} />
        </label>
        <button className="primary-btn" type="submit">
          Generate Recommendation
        </button>
      </form>
    </section>
  );
}

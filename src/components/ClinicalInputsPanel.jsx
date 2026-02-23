const COMORBIDITY_OPTIONS = [
  "Hypertension",
  "Diabetes mellitus",
  "Coronary artery disease",
  "Chronic kidney disease",
  "COPD",
  "Prior stroke/TIA",
  "Anticoagulation use",
];

export default function ClinicalInputsPanel({
  form,
  onChange,
  onSubmit,
  psaDensity,
  predictedProstateVolumeMl,
  onSaveCsv,
  patientId,
}) {
  return (
    <section className="card form">
      <div className="card-header">
        <h2>Clinical Inputs</h2>
        <p>Capture TNM staging and key variables for patient {patientId ?? "N/A"}.</p>
      </div>
      <form onSubmit={onSubmit} className="input-grid">
        <div className="form-section">
          <h3>Staging</h3>
          <div className="section-grid">
            <label>
              cT stage
              <select name="tStage" value={form.tStage} onChange={onChange}>
                <option>T0</option>
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
          </div>
        </div>

        <div className="form-section">
          <h3>Basic Clinical History</h3>
          <div className="section-grid">
            <label>
              Age
              <input name="age" value={form.age} onChange={onChange} />
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
              Family history
              <select name="familyHistory" value={form.familyHistory} onChange={onChange}>
                <option>None known</option>
                <option>First-degree relative</option>
                <option>Multiple affected relatives</option>
              </select>
            </label>
            <label>
              Lower urinary tract symptoms
              <select
                name="lowerUrinarySymptoms"
                value={form.lowerUrinarySymptoms}
                onChange={onChange}
              >
                <option>None</option>
                <option>Mild</option>
                <option>Moderate</option>
                <option>Severe</option>
              </select>
            </label>
            <label>
              Prior biopsy
              <select name="priorBiopsy" value={form.priorBiopsy} onChange={onChange}>
                <option>No prior biopsy</option>
                <option>Prior negative biopsy</option>
                <option>Prior positive biopsy</option>
              </select>
            </label>
            <label>
              Predicted prostate volume (mL)
              <input value={predictedProstateVolumeMl} readOnly />
            </label>
            <label>
              PSA density (PSA/volume)
              <input value={psaDensity || "n/a"} readOnly />
            </label>
          </div>
        </div>

        <div className="form-section">
          <h3>Relevant Comorbidities</h3>
          <div className="checkbox-grid">
            {COMORBIDITY_OPTIONS.map((option) => (
              <label key={option} className="checkbox-label">
                <input
                  type="checkbox"
                  name="comorbidities"
                  value={option}
                  checked={form.comorbidities.includes(option)}
                  onChange={onChange}
                />
                {option}
              </label>
            ))}
          </div>
        </div>

        <div className="form-section">
          <h3>Patient Preferences</h3>
          <div className="section-grid">
            <label>
              Sexual function preservation
              <select
                name="preferencesSexualFunction"
                value={form.preferencesSexualFunction}
                onChange={onChange}
              >
                <option>High priority</option>
                <option>Moderate priority</option>
                <option>Low priority</option>
              </select>
            </label>
            <label>
              Urinary continence preservation
              <select
                name="preferencesUrinaryContinence"
                value={form.preferencesUrinaryContinence}
                onChange={onChange}
              >
                <option>High priority</option>
                <option>Moderate priority</option>
                <option>Low priority</option>
              </select>
            </label>
            <label>
              Treatment intensity preference
              <select
                name="preferencesTreatmentIntensity"
                value={form.preferencesTreatmentIntensity}
                onChange={onChange}
              >
                <option>Aggressive treatment</option>
                <option>Balanced</option>
                <option>Minimize treatment burden</option>
              </select>
            </label>
            <label>
              Follow-up burden tolerance
              <select
                name="preferencesFollowUpBurden"
                value={form.preferencesFollowUpBurden}
                onChange={onChange}
              >
                <option>High</option>
                <option>Moderate</option>
                <option>Low</option>
              </select>
            </label>
            <label className="full-width">
              Additional priorities
              <input
                name="additionalPreferences"
                value={form.additionalPreferences}
                onChange={onChange}
                placeholder="e.g. travel constraints, work duties, caregiver role"
              />
            </label>
          </div>
        </div>

        <button className="primary-btn" type="submit">
          Generate Recommendation
        </button>
        <button className="ghost-btn secondary-btn" type="button" onClick={onSaveCsv}>
          Save
        </button>
      </form>
    </section>
  );
}

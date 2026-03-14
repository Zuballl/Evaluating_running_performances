import { useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const ALL_TRAINED_FEATURES_COUNT = 10;
const PREPROCESSING_MODE = 'standard_clip';

const FIELD_LABELS = {
  age: 'Age',
  is_male: 'Gender (1 = male, 0 = female)',
  athlete_weight: 'Weight [kg]',
};

const FEATURE_DISPLAY = {
  pace_min_km: 'Pace [min/km]',
  average_speed: 'Average Speed [km/h]',
  athlete_weight: 'Athlete Weight [kg]',
  total_distance: 'Total Distance [km]',
  average_hr: 'Average HR [bpm]',
  final_cadence: 'Final Cadence [spm]',
  aerobic_decoupling: 'Aerobic Decoupling [%]',
  age: 'Age [years]',
  is_male: 'Sex (Male=1, Female=0)',
  elevation_gain: 'Elevation Gain [m]',
};

const FEATURE_HELP = {
  pace_min_km: 'Average minutes needed to run 1 km. Lower is faster.',
  average_speed: 'Average speed during activity in kilometers per hour.',
  athlete_weight: 'Body mass used in the model context (kilograms).',
  total_distance: 'Total covered distance in kilometers.',
  average_hr: 'Average heart rate in beats per minute.',
  final_cadence: 'Average running cadence in steps per minute.',
  aerobic_decoupling: 'Change of speed-to-heart-rate efficiency over time.',
  age: 'Runner age in years.',
  is_male: 'Binary sex indicator from the dataset (1 male, 0 female).',
  elevation_gain: 'Total ascent in meters.',
};

function percentileBand(percentile) {
  if (percentile >= 80) return 'very strong performance profile';
  if (percentile >= 60) return 'strong performance profile';
  if (percentile >= 40) return 'balanced performance profile';
  if (percentile >= 20) return 'developing performance profile';
  return 'early-stage performance profile';
}

function toReadableFeature(name) {
  return FEATURE_DISPLAY[name] ?? name.replaceAll('_', ' ');
}

function App() {
  const [fitFile, setFitFile] = useState(null);
  const [missingFields, setMissingFields] = useState([]);
  const [fallbackValues, setFallbackValues] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const canSubmit = useMemo(() => {
    if (!fitFile) return false;
    if (missingFields.length === 0) return true;
    return missingFields.every((field) => fallbackValues[field] !== undefined && fallbackValues[field] !== '');
  }, [fitFile, missingFields, fallbackValues]);

  const interpretation = useMemo(() => {
    if (!result) return null;
    const topThree = result.top_contributions.slice(0, 3);
    const dominant = topThree.map((item) => toReadableFeature(item.feature)).join(', ');
    return {
      band: percentileBand(result.percentile),
      dominant,
    };
  }, [result]);

  const handleUpload = async () => {
    if (!fitFile) {
      setError('Choose a FIT file first.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('fit_file', fitFile);

      for (const field of missingFields) {
        const value = fallbackValues[field];
        if (value !== undefined && value !== '') {
          formData.append(field, String(value));
        }
      }

      const response = await fetch(
        `${API_BASE}/predict-fit?top_k=${ALL_TRAINED_FEATURES_COUNT}&preprocessing=${PREPROCESSING_MODE}`,
        {
        method: 'POST',
        body: formData,
        }
      );

      const payload = await response.json();

      if (!response.ok) {
        const detail = payload?.detail;
        if (detail?.code === 'missing_profile_fields') {
          setMissingFields(detail.missing_fields ?? []);
          setError('Please fill missing profile fields and submit again.');
          setResult(null);
          return;
        }
        throw new Error(detail?.message || detail || 'Prediction failed');
      }

      setResult(payload);
      setMissingFields([]);
      setError('');
    } catch (err) {
      setResult(null);
      setError(err.message || 'Unexpected error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-shell">
      <div className="aurora" aria-hidden="true" />
      <main className="card">
        <header className="hero">
          <p className="eyebrow">Running Performance</p>
          <h1>PCA Score + XAI</h1>
          <p className="subtitle">
            Upload a Garmin FIT file and get score, percentile, and feature contributions.
          </p>
        </header>

        <section className="panel">
          <label className="file-input-label">
            FIT file
            <input
              type="file"
              accept=".fit"
              onChange={(e) => setFitFile(e.target.files?.[0] ?? null)}
            />
          </label>

          {missingFields.length > 0 && (
            <div className="missing-grid">
              <h3>Missing profile fields</h3>
              {missingFields.map((field) => (
                <label key={field}>
                  {FIELD_LABELS[field] ?? field}
                  <input
                    type="number"
                    step={field === 'athlete_weight' ? '0.1' : '1'}
                    value={fallbackValues[field] ?? ''}
                    onChange={(e) =>
                      setFallbackValues((prev) => ({
                        ...prev,
                        [field]: e.target.value,
                      }))
                    }
                  />
                </label>
              ))}
            </div>
          )}

          <button disabled={!canSubmit || loading} onClick={handleUpload}>
            {loading ? 'Scoring...' : 'Score this activity'}
          </button>
        </section>

        {error && <p className="error-box">{error}</p>}

        {result && (
          <section className="result-panel">
            <div className="score-strip">
              <div>
                <p className="metric-label">Score</p>
                <p className="metric-value">{result.score.toFixed(3)}</p>
              </div>
              <div>
                <p className="metric-label">Percentile</p>
                <p className="metric-value">{result.percentile.toFixed(2)}%</p>
              </div>
            </div>

            {interpretation && (
              <div className="insight-box">
                <p>
                  This activity maps to a <strong>{interpretation.band}</strong>. The strongest drivers were{' '}
                  <strong>{interpretation.dominant}</strong>.
                </p>
              </div>
            )}

            <h3>All feature contributions</h3>
            <ul className="contrib-list">
              {result.top_contributions.map((item) => (
                <li key={item.feature}>
                  <div className="contrib-row-head">
                    <span className="feature-label" title={FEATURE_HELP[item.feature] ?? ''}>
                      {toReadableFeature(item.feature)}
                    </span>
                    <span>{item.contribution_pct_abs.toFixed(2)}%</span>
                  </div>
                  <div className="contrib-bar-track" role="img" aria-label={`${item.feature} contribution bar`}>
                    <div
                      className={`contrib-bar ${item.contribution >= 0 ? 'positive' : 'negative'}`}
                      style={{ width: `${Math.max(item.contribution_pct_abs, 2)}%` }}
                    />
                  </div>
                </li>
              ))}
            </ul>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;

"use strict";
const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = Number(process.env.PORT || 3000);
const DATA_FILE = process.env.DATA_FILE || path.join(__dirname, "data.json");
const MAX_HISTORY = 5000;
const RF_MODEL_FILE = process.env.RF_MODEL_FILE || path.join(__dirname, "model", "random_forest.json");
let rfModel = null;
try { if (fs.existsSync(RF_MODEL_FILE)) rfModel = JSON.parse(fs.readFileSync(RF_MODEL_FILE, "utf8")); } catch (e) { console.error("Could not load Random Forest model:", e.message); }
const RF_FEATURES = ["s1_pct", "s2_pct", "vibration_score_10", "dynamic_accel", "accel_magnitude", "pitch_deg", "roll_deg", "mic_peak_to_peak_adc", "rain_detected", "bmp_pressure", "dht_temp", "dht_humidity"];
function rfPredict(d) {
  if (!rfModel || !Array.isArray(rfModel.trees) || !rfModel.trees.length) return null;
  const values = RF_FEATURES.map((k, i) => { const raw = k === "rain_detected" ? (d[k] === true ? 1 : d[k] === false ? 0 : null) : firstNumber(d[k]); return raw === null ? Number(rfModel.medians[i] || 0) : raw; });
  const votes = {};
  for (const tree of rfModel.trees) { let node = 0, steps = 0; while (tree.children_left[node] !== tree.children_right[node] && steps++ < 1000) { const f = tree.feature[node]; node = values[f] <= tree.threshold[node] ? tree.children_left[node] : tree.children_right[node]; } const counts = tree.value[node]; let best = 0; for (let i = 1; i < counts.length; i++) if (counts[i] > counts[best]) best = i; const label = rfModel.classes[best]; votes[label] = (votes[label] || 0) + 1; }
  const label = Object.keys(votes).sort((a,b) => votes[b]-votes[a])[0]; return { risk: label, rf_vote_share_pct: Math.round(votes[label] / rfModel.trees.length * 100), rf_votes: votes, model_name: "RandomForestClassifier", model_version: rfModel.version || "1.0" };
}
const SMS_COOLDOWN_MS = Math.max(60_000, Number(process.env.SMS_COOLDOWN_MS || 30 * 60 * 1000));
const SMS_ALERT_SCORE = clamp(Number(process.env.SMS_ALERT_SCORE || 67), 0, 100);
let lastSmsAt = 0;
let smsInFlight = false;
app.use(cors({ origin: true }));
app.use(express.json({ limit: "64kb" }));
app.use(express.static(path.join(__dirname, "public")));

let history = [];
try {
  if (fs.existsSync(DATA_FILE)) {
    const parsed = JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
    if (Array.isArray(parsed)) history = parsed.slice(-MAX_HISTORY);
  }
} catch (e) { console.error("Could not load data file:", e.message); }
function persist() {
  try {
    fs.mkdirSync(path.dirname(DATA_FILE), { recursive: true });
    const tmp = DATA_FILE + ".tmp";
    fs.writeFileSync(tmp, JSON.stringify(history));
    fs.renameSync(tmp, DATA_FILE);
  } catch (e) { console.error("Persistence error:", e.message); }
}
function finite(v) { return typeof v === "number" && Number.isFinite(v) ? v : null; }
function firstNumber(...values) { for (const v of values) { const n = typeof v === "number" ? v : (v === null || v === undefined || v === "" ? NaN : Number(v)); if (Number.isFinite(n)) return n; } return null; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function derivePrototypeRisk(d) {
  // A transparent DEMONSTRATION index only, not an ML prediction or validated probability.
  const parts = [];
  const vibration = firstNumber(d.vibration_score_10, d.vibration_score, d.vibration);
  if (vibration !== null) parts.push(clamp(vibration / 10, 0, 1) * 40);
  const soils = [firstNumber(d.s1_pct, d.moisture), firstNumber(d.s2_pct, d.moisture2)].filter(v => v !== null);
  if (soils.length) parts.push(clamp(soils.reduce((a,b) => a+b, 0) / soils.length, 0, 100) * 0.30);
  const rain = d.rain_detected !== undefined ? d.rain_detected : d.rain;
  if (typeof rain === "boolean" || rain === 0 || rain === 1) parts.push((rain === true || rain === 1) ? 15 : 0);
  const mic = firstNumber(d.mic_peak_to_peak_adc, d.mic_peak_to_peak);
  if (mic !== null) parts.push(clamp(mic / 1200, 0, 1) * 15);
  if (!parts.length) return { risk: "UNKNOWN", risk_score: null, prototype_risk_index: null };
  const score = Math.round(clamp(parts.reduce((a,b) => a+b, 0), 0, 100));
  return { risk: score >= 67 ? "HIGH" : score >= 34 ? "MEDIUM" : "LOW", risk_score: score, prototype_risk_index: score };
}
function enrich(d) {
  const r = derivePrototypeRisk(d);
  const rf = rfPredict(d);
  return {
    ...d,
    // Keep exact incoming ESP32 field names and add aliases the existing dashboard reads.
    ax: firstNumber(d.ax, d.accel_x), ay: firstNumber(d.ay, d.accel_y), az: firstNumber(d.az, d.accel_z),
    accel_magnitude: firstNumber(d.accel_magnitude, d.magnitude),
    dynamic_accel: firstNumber(d.dynamic_accel),
    pitch_deg: firstNumber(d.pitch_deg, d.pitch), roll_deg: firstNumber(d.roll_deg, d.roll),
    s1_pct: firstNumber(d.s1_pct, d.moisture), s2_pct: firstNumber(d.s2_pct, d.moisture2),
    rain_detected: typeof d.rain_detected === "boolean" ? d.rain_detected : Boolean(d.rain),
    ...r,
    ...(rf ? { risk: rf.risk, random_forest: rf, risk_source: "trained_random_forest" } : { risk_source: "prototype_heuristic_not_validated" }),
    // A class vote share is not a calibrated landslide probability.
    landslide_probability_pct: null, confidence: null,
    backend_status: rf ? "random_forest_model_loaded_requires_validation" : "no_trained_random_forest_model_loaded"
  };
}
async function sendSmsAlert(record) {
  const authkey = process.env.MSG91_AUTHKEY;
  const templateId = process.env.MSG91_TEMPLATE_ID;
  const recipients = (process.env.SMS_RECIPIENTS || "").split(",").map(x => x.trim()).filter(Boolean);
  if (!authkey || !templateId || recipients.length === 0) {
    console.warn("[PRAHARI SMS] Alert condition met, but SMS is not configured (MSG91_AUTHKEY, MSG91_TEMPLATE_ID, SMS_RECIPIENTS). No SMS sent.");
    return { sent: false, reason: "sms_not_configured" };
  }
  if (smsInFlight) return { sent: false, reason: "sms_in_flight" };
  if (Date.now() - lastSmsAt < SMS_COOLDOWN_MS) return { sent: false, reason: "cooldown" };
  smsInFlight = true;
  try {
    // Indian SMS delivery requires an approved DLT template. Configure its variables as:
    // VAR1 = risk label, VAR2 = risk index, VAR3 = device ID, VAR4 = dashboard URL.
    const recipientsPayload = recipients.map(mobiles => ({
      mobiles: mobiles.replace(/\D/g, ""),
      VAR1: String(record.risk || "HIGH"),
      VAR2: String(record.risk_score ?? "NA"),
      VAR3: String(record.device_id || "PRAHARI device"),
      VAR4: process.env.PUBLIC_DASHBOARD_URL || "https://prahari.space"
    }));
    const response = await fetch("https://control.msg91.com/api/v5/flow", {
      method: "POST",
      headers: { "authkey": authkey, "accept": "application/json", "content-type": "application/json" },
      body: JSON.stringify({ template_id: templateId, short_url: "0", recipients: recipientsPayload }),
      signal: AbortSignal.timeout(10000)
    });
    const body = await response.text();
    if (!response.ok) {
      console.error(`[PRAHARI SMS] MSG91 HTTP ${response.status}: ${body.slice(0, 500)}`);
      return { sent: false, reason: "provider_error" };
    }
    lastSmsAt = Date.now();
    console.log(`[PRAHARI SMS] Provider accepted alert for ${recipients.length} recipient(s).`);
    return { sent: true };
  } catch (e) {
    console.error("[PRAHARI SMS] Send failed:", e.message);
    return { sent: false, reason: "send_failed" };
  } finally {
    smsInFlight = false;
  }
}

app.get("/", (_req, res) => res.json({ service: "PRAHARI API", status: "ok", endpoints: ["POST /data", "GET /data", "GET /api/latest", "GET /api/history?limit=40", "GET /health"], random_forest_loaded: Boolean(rfModel), random_forest_classes: rfModel ? rfModel.classes : [] }));
app.get("/health", (_req, res) => res.json({ ok: true, stored_samples: history.length, latest_at: history.length ? history[history.length - 1].received_at : null, random_forest_loaded: Boolean(rfModel), random_forest_classes: rfModel ? rfModel.classes : [] }));
app.post("/data", (req, res) => {
  if (!req.is("application/json")) return res.status(415).json({ success: false, error: "Send Content-Type: application/json" });
  if (!req.body || typeof req.body !== "object" || Array.isArray(req.body)) return res.status(400).json({ success: false, error: "JSON object required" });
  const d = req.body;
  const record = { ...enrich(d), received_at: new Date().toISOString() };
  history.push(record);
  if (history.length > MAX_HISTORY) history = history.slice(-MAX_HISTORY);
  persist();
  console.log(`[PRAHARI] received device=${record.device_id || "unknown"} sample=${record.sample_id ?? "n/a"} index=${record.risk_score ?? "N/A"}`);
  if ((record.random_forest && record.random_forest.risk === "HIGH") || (!record.random_forest && record.risk_score !== null && record.risk_score >= SMS_ALERT_SCORE)) {
    void sendSmsAlert(record);
  }
  res.status(201).json({ success: true, message: "Sensor sample received", device_id: record.device_id || null, sample_id: record.sample_id ?? null, received_at: record.received_at, risk: record.risk, risk_score: record.risk_score, prototype_risk_index: record.prototype_risk_index, landslide_probability_pct: null, confidence: null, note: rfModel ? "Random Forest class output; not a validated landslide probability or operational warning." : "No trained Random Forest model loaded. Prototype index only; train and validate a model before interpreting risk classes." });
});
app.get("/data", (_req, res) => {
  if (!history.length) return res.status(404).json({ success: false, message: "No sensor data received yet" });
  res.json({ success: true, latest: history[history.length - 1] });
});
app.get("/api/latest", (_req, res) => {
  const latest = history.length ? history[history.length - 1] : null;
  res.json({ success: true, ok: true, latest, data: latest });
});
app.get("/api/history", (req, res) => {
  const parsed = Number.parseInt(req.query.limit || "40", 10);
  const limit = clamp(Number.isFinite(parsed) ? parsed : 40, 1, 500);
  // Oldest to newest, matching the existing frontend's `latest = readings[last]` assumption.
  const data = history.slice(-limit);
  res.json({ success: true, count: data.length, data });
});
app.use((err, _req, res, _next) => { console.error(err.stack || err); res.status(500).json({ success: false, error: "Internal server error" }); });
app.listen(PORT, "0.0.0.0", () => console.log(`PRAHARI API listening on port ${PORT}`));

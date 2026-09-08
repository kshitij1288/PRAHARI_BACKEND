# PRAHARI — Wayanad Backend v2

This rebuild is specifically aligned to the **30 July 2024 Wayanad (Mundakkai–Chooralmala) landslide case study**.

## Important scientific distinction

This package contains TWO different datasets:

1. `data/wayanad_case_study.csv`
   - Case-study facts compiled from official/public sources.
   - Includes the 2024 event and documented historical Wayanad landslides.
   - It is **not** used as fake ESP32 sensor readings.

2. `data/wayanad_sensor_prototype.csv`
   - Generated prototype sensor scenarios using Wayanad-relevant risk relationships.
   - It lets the website and ESP32 pipeline run end-to-end.
   - It is **not measured 2024 Wayanad sensor data** and its model accuracy must not be presented as field validation.

For a research-grade model, replace the prototype sensor CSV with measured, labelled field observations.

## Official/source basis

- Kerala State Disaster Management Authority — 2024 Wayanad landslide reports / PDNA:
  https://sdma.kerala.gov.in/reports-landslides-2024/
- IMD — Heavy rainfall during 29–30 July 2024 and landslides in Kerala:
  https://mausam.imd.gov.in/imd_latest/monsoonreport2024.pdf
- Geological Survey of India — First Information Report, Mundakkai–Chooralmala:
  https://bhusanket.gsi.gov.in/Output/LS_Incidence_Report/2024/First%20Information%20Report%20on%20Wayanad%20debris%20flows%20July%202024.pdf
- KSDMA / GSI Wayanad susceptibility map:
  https://sdma.kerala.gov.in/wp-content/uploads/2025/06/Landslide_Susceptibility_Zones_Districts_FINAL.pdf
- NRSC/NDEM Wayanad impact products:
  https://ndem.nrsc.gov.in/

## API

- `POST /data` — ESP32-compatible inference + live CSV append
- `POST /api/data` — same
- `GET /api/latest`
- `GET /api/history?limit=100`
- `GET /api/predict?...`
- `GET /api/status`
- `GET /api/export.csv`

## ESP32 JSON

The backend accepts:
`s1_pct, s2_pct, dht_temp, dht_humidity, bmp_pressure, rain_detected, vibration, magnitude, pitch, roll`

The response contains:
`risk, confidence, risk_code, probabilities, sample_id, timestamp`

## Hostinger

Keep the existing frontend and deploy this Node/Express backend separately, ideally as:
`api.yourdomain.com`

Then change the ESP32 server endpoint from the local IP to:
`https://api.yourdomain.com/data`

Use HTTPS for production.

## Local run

```bash
npm install
npm start
```

The server defaults to port 5000.

## Retraining

Install Python requirements:

```bash
pip install -r requirements-training.txt
```

Then:

```bash
python train_model.py
```

The included training script is a starting point; it should be upgraded when real labelled sensor observations are available.



# Sensor Anomaly Detection Results V3

This report evaluates the sensor anomaly-detection pipeline from three perspectives:

1. **Segment reliability** — how consistently each sensor mounting period could be monitored.
2. **Real-data findings** — sudden tilt events and slow drift detected in approximately 30 days of real sensor data.
3. **Fault-detection stress test** — how well the detector catches artificially injected faults.

> **Important:** Detected events are **candidate anomalies**, not confirmed structural problems. A sudden change can also be caused by sensor re-mounting, cable movement, accidental contact, or other non-structural activity. Real events should therefore be cross-checked with field/maintenance records.

---

## Table 1 — Segment Reliability

A **segment** represents one physical mounting period for one sensor.

For example, `RFM_0004_s3` means sensor `RFM_0004` during its **third mounting period**.

| Column             | Meaning                                                                                                                                                         |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `segment`          | One physical mounting period for one sensor.                                                                                                                    |
| `sessions`         | Number of separate sensor on-times, power-ups, or connections during that mounting period.                                                                      |
| `monitored`        | Number of sessions that collected enough data for the anomaly detector to monitor them.                                                                         |
| `unarmed/settling` | Sessions that could not yet be checked because the detector was still collecting reference data (`settling`) or could not obtain a stable baseline (`unarmed`). |
| `events`           | Number of sudden tilt **steps** detected in the segment.                                                                                                        |
| `drift_sessions`   | Sessions flagged for a slow, continuous change in tilt rather than a sudden step.                                                                               |
| `sigma_q_deg`      | Estimated natural baseline noise of the sensor, in degrees. Higher values mean the sensor is naturally more noisy.                                              |
| `low_sensitivity`  | `True` means the sensor is noisy enough that small real tilts may be hidden by its natural noise.                                                               |

### Simple interpretation

> This table tells us, for each mounting period, how much of the time we could actually trust the sensor to tell us if something moved, and how noisy the sensor naturally is.

---

## Table 2 — Events / Drift on Real Data

This table contains findings from **real sensor data** covering approximately 30 days.

These are **not simulated faults**.

| Type       | Meaning                                                                                                            |
| ---------- | ------------------------------------------------------------------------------------------------------------------ |
| `EVENT`    | A sudden tilt step was detected. `shift_deg` gives the estimated size of the sudden change in degrees.             |
| `DRIFT`    | A slow, continuous tilt trend was detected. `drift_deg_day` gives the estimated rate of change in degrees per day. |
| `when_IST` | Time of the detected event in Indian Standard Time (IST).                                                          |

### Important caveat

These are **candidate anomalies**, not confirmed structural problems.

A detected sudden change can potentially be caused by:

* Actual structural movement
* Sensor re-mounting
* Cable movement
* Someone touching or disturbing the sensor
* Maintenance activity
* Installation changes
* Other environmental or operational activity

Therefore, detected events should be cross-checked with **field/maintenance logs** before being treated as actual structural movement.

---

## Table 3 — Fault Detection Stress Test

This table evaluates the detector using **synthetic faults injected into clean sections of real sensor data**.

The process is:

1. Select a relatively clean section of real sensor data.
2. Artificially inject a known fault.
3. Run the anomaly detector.
4. Check whether the detector catches the fault.
5. Measure the detection delay.

| Column           | Meaning                                                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `case`           | Type and size of the artificially injected fault. Example: `step 0.3` means a sudden artificial tilt change of `0.3°`. |
| `n`              | Number of independent test runs for that fault type.                                                                   |
| `detect_rate`    | Fraction of test runs in which the detector successfully detected the injected fault.                                  |
| `median_delay_h` | Median time, in hours, between the start of the injected fault and its detection.                                      |

### Example

```text
case = step 0.3
n = 20
detect_rate = 0.95
median_delay_h = 1.2
```

This means:

> A synthetic sudden tilt of `0.3°` was injected into 20 independent tests. The detector caught it in 95% of the tests, with a median detection delay of 1.2 hours.

This does **not** mean that every real-world `0.3°` structural movement will be detected in exactly 1.2 hours. Real performance depends on sensor noise, data availability, mounting conditions, and the characteristics of the actual event.

---

## How to Read the Three Tables Together

| Table                     | Main Question                                                                 |
| ------------------------- | ----------------------------------------------------------------------------- |
| **Table 1 — Segments**    | Can we trust this sensor/mounting period enough to monitor it?                |
| **Table 2 — Real Events** | What potentially unusual behaviour occurred in the real data?                 |
| **Table 3 — Stress Test** | If a known fault occurs, how reliably and quickly does the detector catch it? |



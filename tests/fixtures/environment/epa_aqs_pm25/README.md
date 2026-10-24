# EPA AirData daily PM2.5 (parameter 88101), 2024-07-05 and 2024-09-09

`daily_88101_2024_0705_0909.csv` holds every row of EPA's pre-generated
AirData file `daily_88101_2024.zip`
(https://aqs.epa.gov/aqsweb/airdata/download_files.html) whose `Date Local`
is 2024-07-05 or 2024-09-09, retrieved 2026-10-23 (upstream Last-Modified
2026-07-13). The rows were chosen by date alone, before any comparison with
the detector. Licence: a U.S. Government work, in the public domain.

Kept columns, values unchanged: `State Code`, `County Code`, `Site Num`,
`POC`, `Latitude`, `Longitude`, `Sample Duration`, `Pollutant Standard`,
`Date Local`, `Event Type`, `Observation Count`, `Arithmetic Mean`, `AQI`.
4,602 rows; 2,715 monitor-days have a 24-hour standard value as measured
(`Pollutant Standard` set, `Event Type` not `Excluded`).

| File | SHA-256 |
|---|---|
| upstream `daily_88101_2024.zip` | `409c2d884935bd0a4099d640b7c526ecb0af1d3aac049697e34c2f9f69dbe91d` |
| `daily_88101_2024_0705_0909.csv` | `d924c6d21baf40d879e0467fb825dc61eafe68207dc2fc3828f5e85c76fe72d0` |

Consumed by `tests/detectors/test_air_quality_recorded.py`.

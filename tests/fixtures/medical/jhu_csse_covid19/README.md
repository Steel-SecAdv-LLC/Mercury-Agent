# JHU CSSE COVID-19 confirmed cases: Italy, 2020-02-15 to 2020-07-15

`italy_confirmed_2020.csv` is an extract of
`csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv`
from the COVID-19 Data Repository by the Center for Systems Science and
Engineering at Johns Hopkins University
(https://github.com/CSSEGISandData/COVID-19, archived 2023), retrieved
2026-10-23. Licence: Creative Commons Attribution 4.0 International,
copyright Johns Hopkins University 2020.

Cite: Dong E, Du H, Gardner L. An interactive web-based dashboard to track
COVID-19 in real time. Lancet Infect Dis 20(5):533-534 (2020).

Extraction: the row with `Country/Region` = `Italy` and an empty
`Province/State`, columns 2020-02-15 to 2020-07-15 inclusive, written as
`date` (ISO 8601) and `cumulative_confirmed` (unchanged values). Daily new
cases are the differences of consecutive days; 2020-06-19 is negative (-148),
a reporting correction.

| File | SHA-256 |
|---|---|
| upstream `time_series_covid19_confirmed_global.csv` | `e6234a59eec4359d2577358b5220e1a7e3da74c162913cdb7d882db1413f98c2` |
| `italy_confirmed_2020.csv` | `d77712c6f75cb8f428f042e57147350c5382b3ead460edf03810a17b0bb5c341` |

Consumed by `tests/medical/test_case_surge_recorded.py`.

# The watch

Mercury reading its live feeds on its own and saying what matters.

```bash
mercury-agent watch --once     # read every feed once and say what was seen
mercury-agent watch            # keep watching; new findings are printed as they appear
mercury-agent watch --status   # what the watch last saw, without reading anything
```

A server runs the same watch in the background when `MERCURY_WATCH_ENABLED=true`
and shows it at `/watch` (with `MERCURY_FRONTEND_ENABLED=true`) and at
`GET /api/v1/watch`.

## What it reads

| Feed | Source | Read every | A finding is |
|------|--------|-----------|--------------|
| `earthquakes` | USGS, earthquakes of the past hour | 5 minutes | magnitude 4.5 or above; significant from 6.0 (strong), severe from 7.0 (major), by USGS magnitude class |
| `space_weather` | NOAA SWPC, planetary Kp and solar wind | 15 minutes | Kp 5 or above: a geomagnetic storm on the NOAA G scale; significant from G3, severe from G4 |
| `tsunami_buoys` | NOAA NDBC, DART buoys | 15 minutes | a buoy in event mode in the past three hours (its tsunameter switched to rapid sampling); significant when two or more are |
| `river_gauges` | USGS Water Services | 15 minutes | a gauge above every level the watch has recorded for it, once it holds two days of readings |
| `fema_declarations` | OpenFEMA, tropical-cyclone declarations | 1 hour | declarations in the past three days; significant in the busiest week of the year |

## How it decides

Scoring a batch against itself flags about one row in twenty whatever the batch
holds: on an ordinary day that was 1,036 buoy readings and 99 gauge readings.
The watch does not do that. Each feed uses the rule its field already uses
where one exists, and where the source carries none (a river gauge's flood
stage is not in the USGS reading) the plainest true statement available: this
series has passed everything recorded for it. Most reads report nothing, and
the watch says so, with one line per feed.

A finding is reported once. While its condition lasts it is the same finding;
when a read no longer shows it, it ends; if it becomes more severe it is
reported again. Each finding states its source, the rule that made it a
finding, and what it cannot tell: a preliminary magnitude, a buoy that may be
running a test, a gauge whose flood stage Mercury does not have.

## What it keeps

One SQLite file under `MERCURY_WATCH_DIR` (default `~/.mercury/watch`): up to
1,000 readings per series, every finding with when it was first and last seen,
and each feed's last status. The watch gives each loader its own cache with a
lifetime shorter than the feed's cadence, so every read reaches the source;
the loaders' default is an hour, and a day for the buoys.

## What it does not do

- It does not issue warnings. Tsunami warnings come from the NOAA tsunami
  warning centers and flood warnings from the National Weather Service.
- It does not notify anyone. Findings are printed, stored and served; sending
  them somewhere is the deployment's choice.
- It reads five feeds. The other loaders are either not live (benchmark files,
  series that end in 2024), need a key, or have no rule yet that would say
  something true and useful.
- A feed that cannot be read is shown as down and tried again at its cadence.

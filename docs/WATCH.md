# The watch

Mercury reading its live feeds on its own and saying what matters.

```bash
mercury-agent watch --once     # read every feed once and say what was seen
mercury-agent watch            # keep watching; new findings are printed as they appear
mercury-agent watch --status   # what the watch last saw, without reading anything
```

A server runs the same watch in the background when `MERCURY_WATCH_ENABLED=true`
and serves it at `GET /api/v1/watch`. With `MERCURY_FRONTEND_ENABLED=true` it is
shown at `/watch` and in the console (`/console`, Mercury Agent, WATCH tab; or
"Open the watch" in the command palette).

## What it reads

| Feed | Source | Read every | A finding is |
|------|--------|-----------|--------------|
| `earthquakes` | USGS, earthquakes of the past hour | 5 minutes | magnitude 4.5 or above; significant from 6.0 (strong), severe from 7.0 (major), by USGS magnitude class |
| `space_weather` | NOAA SWPC, planetary Kp and solar wind | 15 minutes | Kp 5 or above: a geomagnetic storm on the NOAA G scale; significant from G3, severe from G4 |
| `tsunami_buoys` | NOAA NDBC, DART buoys | 15 minutes | a buoy in event mode in the past three hours (its tsunameter switched to rapid sampling); significant when two or more are |
| `river_gauges` | NOAA National Water Prediction Service, every NWS river gauge (about 12,000) | 30 minutes | a gauge whose latest reading the NWS classes as minor flooding or above; significant at moderate, severe at major. Action stage is counted, not reported |
| `volcanoes` | USGS Hazard Notification System, every monitored U.S. volcano | 30 minutes | a volcano above NORMAL on the USGS volcano alert levels: ADVISORY, significant at WATCH, severe at WARNING |
| `fema_declarations` | OpenFEMA, tropical-cyclone declarations | 1 hour | declarations in the past three days; significant in the busiest week of the year |

## How it decides

Scoring a batch against itself flags about one row in twenty whatever the batch
holds: on an ordinary day that was 1,036 buoy readings and 99 gauge readings.
The watch does not do that. Each feed uses the rule its field already uses:
USGS magnitude classes, the NOAA G scale, a buoy's own event mode, the flood
category a forecast office sets for its gauge, the USGS volcano alert levels. Most reads report nothing, and
the watch says so, with one line per feed.

A finding is reported once. While its condition lasts it is the same finding;
when a read no longer shows it, it ends; if it becomes more severe it is
reported again, and checked again. Each finding states its source, the rule that made it a
finding, and what it cannot tell: a preliminary magnitude, a buoy that may be
running a test.

## How it checks itself

Each finding is looked up again in the record that settles it, and the answer
is kept with the finding:

| Feed | Later record | First checked | Final |
|------|--------------|---------------|-------|
| `earthquakes` | the USGS catalogue's entry for the event | 2 hours | once a seismologist has reviewed it, or USGS has deleted it |
| `space_weather` | NOAA's Kp for the same interval | 6 hours | two days after the interval |
| `tsunami_buoys` | NOAA NCEI's tsunami event database | 12 hours | when NCEI lists a probable or definite tsunami within a day before the trigger; otherwise after 45 days |
| `river_gauges` | the NWS record of the same gauge | 6 hours | once the flood has ended; otherwise after 7 days |
| `volcanoes` | the observatory notice the level came from | 1 hour | once the notice is read |
| `fema_declarations` | OpenFEMA, read again | 1 day | ten days after the declarations |

The verdict is one of four words. `confirmed`: the record says what was
reported. `revised`: it now says something else, and the finding says what
(USGS moved the magnitude from 6.6 to 6.3). `withdrawn`: the record no longer
holds it (USGS deleted an automatic solution; the NWS corrected a reading to
below flood stage). `unconfirmed`: no record supports it (NCEI lists no
tsunami for a buoy trigger, which also fires on seismic shaking and on
tests). A record that cannot be reached is not a verdict: the finding is
tried again at the next interval.

The track record is the count of each verdict over every finding the watch has
made. It is printed by `mercury-agent watch --status`, returned as
`track_record` by the route, and shown on the page and in the console.

## What it keeps

One SQLite file under `MERCURY_WATCH_DIR` (default `~/.mercury/watch`): every
finding with when it was first and last seen and what the later record said,
and each feed's last status. A finding that ends and recurs is a new finding;
the earlier one keeps its own outcome. The watch gives each loader its own cache with a
lifetime shorter than the feed's cadence, so every read reaches the source;
the loaders' default is an hour, and a day for the buoys.

## What it does not do

- It does not issue warnings. Tsunami warnings come from the NOAA tsunami
  warning centers and flood warnings from the National Weather Service.
- It does not notify anyone. Findings are printed, stored and served; sending
  them somewhere is the deployment's choice.
- It reads six feeds. The other loaders are either not live (benchmark files,
  series that end in 2024), need a key, or have no rule yet that would say
  something true and useful.
- River gauges are the National Weather Service's and volcano alert levels
  the USGS's, so those two feeds cover the United States and its territories.
- A feed that cannot be read is shown as down and tried again at its cadence.

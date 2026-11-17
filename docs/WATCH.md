# The watch

Mercury reading its live feeds on its own and saying what matters.

```bash
mercury-agent watch --once     # read every feed once and say what was seen
mercury-agent watch            # keep watching; new findings are printed as they appear
mercury-agent watch --status   # what the watch last saw, without reading anything

mercury-agent alerts add --email you@example.org --place Home --lat 28.37 --lon -80.87
mercury-agent alerts test --email you@example.org   # prove the message arrives, before it matters
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
| `warnings` | National Weather Service alerts in effect | 5 minutes | every alert the Service rates Extreme, nationwide; and every alert in effect at a subscriber's place. Relayed as issued, with the Service's instruction |
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
| `warnings` | the Service's own record of the alert | 30 minutes | once the record is read |
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

## What to do

Every finding says what a person there should do, in the words of the agency
responsible for that hazard and attributed to it: the National Weather
Service's own instruction on its alerts, word for word; its flood safety
guidance on a river in flood; USGS and Ready.gov on an earthquake; the tsunami
warning centers on a buoy's detection; the FEMA Helpline on a declaration.
Mercury writes no safety advice of its own.

## Who it tells

A finding is delivered to the people it concerns.

- **Subscribers, by email.** `mercury-agent alerts add` subscribes an address,
  usually with a place and a radius (200 km unless given) and the least
  severity wanted. A finding with coordinates goes to subscribers within their
  radius of it. A National Weather Service alert goes to the subscribers at
  whose place the Service says it is in effect. A finding with no place (a
  geomagnetic storm) goes to everyone. A subscriber with no place is told
  everything. Email needs the deployment's SMTP server (`MERCURY_SMTP_HOST`
  and the other `MERCURY_SMTP_*` settings); without one nothing is sent and
  the command, the page and the route all say subscribers cannot be reached.
- **Open consoles.** A new finding appears in the console's alert dock and
  banner as it is made, and is spoken if the console is set to speak.
- **From an account.** With the browser front end on, a signed-in person
  manages this for their own address under "Watch alerts" on their account
  page (`GET`/`POST /api/v1/auth/watch-subscriptions`, `DELETE
  /api/v1/auth/watch-subscriptions/{id}`): the places, how far, how severe.
  The address is always the account's verified one. An account names at most
  `MERCURY_WATCH_MAX_PLACES_PER_ACCOUNT` places (default 5). The
  subscriptions move with the account when its address changes, are in its
  data export, and are removed before the account is deleted.
- **By text message.** An address can have one phone number, texted the
  findings that concern its places at or above a severity its owner chooses
  (significant unless changed). The number is texted nothing until a
  six-digit code sent to it has been typed back, from the account page or
  with `mercury-agent alerts phone --email … --number +1415…` and then
  `--code …`. A code is good for ten minutes and five wrong guesses; an
  address is sent at most one a minute and five a day. Texts go through the
  operator's own Twilio account: `MERCURY_SMS_TWILIO_ACCOUNT_SID`,
  `MERCURY_SMS_TWILIO_AUTH_TOKEN`, and `MERCURY_SMS_FROM` (a number the
  account owns) or `MERCURY_SMS_TWILIO_MESSAGING_SERVICE_SID`. Without them
  no text is attempted and the page and the command say so. A reply of STOP
  is honoured by the provider, and the number is then not texted from here
  either. Deliveries are in the same ledger: once per severity, retried when
  they fail.
- **How many places.** The `warnings` feed asks the Service about each
  distinct place (coordinates to two decimals) on every read, up to 250.
  Accounts cannot add a place beyond that. If the operator's command does,
  the longest-subscribed 250 are checked and the feed's summary says how many
  were not; a place the Service will not answer for is skipped and counted,
  and the nationwide relay goes on either way. A place outside the United
  States is outside the Service's area: it is counted as such, not asked
  about again, and takes none of the 250.
- **One webhook.** `MERCURY_WATCH_WEBHOOK_URL` receives a JSON message
  (`title`, `message`, `text`, `priority`, `tags`, and `topic` when
  `MERCURY_WATCH_WEBHOOK_TOPIC` is set) for findings at significant and above
  (`MERCURY_WATCH_WEBHOOK_MIN_SEVERITY`, 1 to 3). That shape is read by ntfy
  and by Slack-compatible chat hooks, which is how a finding reaches a phone.

Each message says what happened, where and how far from the subscriber's
place, what to do and whose guidance that is, the source and basis, and that
Mercury does not issue warnings. Every delivery is recorded: a finding is
sent to a subscriber once per severity, a failed delivery is tried again on
later passes (five attempts) and counted as failed, never dropped silently.
Delivery works from the findings that stand, so someone who subscribes during
a flood is told of it. A condition (a flood, a storm) stands until a read no
longer shows it, and is told again when it comes back; a single occurrence (an
earthquake) stands for a day. Subscribe only an address whose owner asked for
it.

## One watch per store

Any process that opens the store may run a watch: each worker of
`mercury-agent serve --workers 4` does when `MERCURY_WATCH_ENABLED=true`, and
so does `mercury-agent watch` beside a server. One of them holds the store's
lease and does the reading and the telling; the others stand by and take over
when it stops, or 15 minutes after it last renewed. The store is one SQLite
file, so this holds for processes that share its directory on one host.
Replicas that each have their own directory are each a separate watch with
its own subscribers: run the watch in one place and point `MERCURY_WATCH_DIR`
at it.

## What it keeps

One SQLite file under `MERCURY_WATCH_DIR` (default `~/.mercury/watch`): every
finding with when it was first and last seen and what the later record said,
each feed's last status, the subscribers and the delivery ledger. The file is
readable by its owner only; the route serves counts of subscribers and
deliveries, never an address or a place, and an alert that is there only
because it is in effect at a subscriber's place is sent to that subscriber
and not shown on the page. A finding that ends and recurs is a new finding;
the earlier one keeps its own outcome. The watch gives each loader its own cache with a
lifetime shorter than the feed's cadence, so every read reaches the source;
the loaders' default is an hour, and a day for the buoys.

## What it does not do

- It does not issue warnings. It relays the National Weather Service's, and
  tsunami warnings come from the NOAA tsunami warning centers. It is not a
  replacement for Wireless Emergency Alerts, NOAA Weather Radio or local
  sirens, which reach people without a subscription or a working internet
  connection.
- It does not send text messages or place calls itself; a phone is reached
  through the webhook.
- A subscriber is told of earthquakes within their radius, whatever the
  magnitude: a great earthquake is felt further than 200 km, so someone who
  wants those sets a larger radius.
- It reads seven feeds. The other loaders are either not live (benchmark files,
  series that end in 2024), need a key, or have no rule yet that would say
  something true and useful.
- River gauges and alerts are the National Weather Service's and volcano
  alert levels the USGS's, so those feeds cover the United States and its
  territories. Outside them a subscriber is told of earthquakes, tsunami
  buoys and geomagnetic storms only.
- A feed that cannot be read is shown as down and tried again at its cadence.

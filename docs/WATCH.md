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
| `world_warnings` | National authorities' Common Alerting Protocol alerts, through the WMO Severe Weather Information Centre (50 countries on 2026-11-17; not Japan or India) | 10 minutes | every alert in effect at a subscriber's place, matched by the issuing authority's own polygon; significant when it rates the alert Severe, severe when Extreme. Relayed as issued, with the authority's name and instruction. United States alerts are left to `warnings` |
| `gdacs` | Global Disaster Alert and Coordination System (United Nations and European Commission) | 30 minutes | a current tropical cyclone, flood, volcanic eruption or wildfire GDACS rates Orange (significant) or Red (severe), at the point GDACS gives. Earthquakes are the USGS feed's; a drought has no point a radius can be drawn from |
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
| `world_warnings` | the alert's CAP document at the WMO centre, read again | 30 minutes | two days |
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
  whose place the Service says it is in effect, and an alert from another
  country's authority likewise. An earthquake goes to the places where USGS
  says it was felt, however far (see "What it does not do"). A finding with
  no place (a geomagnetic storm) goes to everyone. A subscriber with no place is told
  everything. Email needs the deployment's SMTP server (`MERCURY_SMTP_HOST`
  and the other `MERCURY_SMTP_*` settings); without one nothing is sent and
  the command, the page and the route all say subscribers cannot be reached.
- **Open consoles.** A new finding appears in the console's alert dock and
  banner as it is made, and is spoken if the console is set to speak. Every
  serving process tells its own consoles from the store, whichever process
  read the feed. An alert that is recorded only because it is in effect at a
  subscriber's place is not shown to consoles or on the public page: it would
  say where a subscriber is. (An Extreme National Weather Service alert is
  reported everywhere regardless, so it is shown.)
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
  `--code …`. A code is good for ten minutes and five wrong guesses. At most
  one code a minute and five a day are sent for an address, and to a number
  whichever addresses ask; removing the number does not start the count
  again. A code the provider could not be reached to send is not counted. A
  verified number goes on being texted until a replacement has been
  confirmed with its own code.
  Texts go through the
  operator's own Twilio account: `MERCURY_SMS_TWILIO_ACCOUNT_SID`,
  `MERCURY_SMS_TWILIO_AUTH_TOKEN`, and `MERCURY_SMS_FROM` (a number the
  account owns) or `MERCURY_SMS_TWILIO_MESSAGING_SERVICE_SID`. Without them
  no text is attempted and the page and the command say so. A reply of STOP
  is honoured by the provider, and the number is then not texted from here
  either. Deliveries are in the same ledger, under the same rules as email.
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
Mercury does not issue warnings. Every delivery is recorded. A finding is
sent to a subscriber once, and again only when it becomes more severe than it
was: a flood falling from major to minor is one message, not three. One
that falls and climbs back to a severity it was told at is told again only
when it has been below it for longer than the feed's episode (26 hours for a
river gauge): a gauge wavering between two stages is one flood. An address
with several places is sent a finding once, whichever place it is nearest,
and removing a place sends nothing again. A warning is one finding across
its updates and across its pieces: the National Weather Service and other
authorities give each update a new identifier and often have one warning in
effect as several messages, each for its own zones (23 of the 527 alerts in
effect on 2026-11-18 were pieces of 8 warnings). The finding follows the
warning, and reaches everyone in any piece. A text is cut to two segments in
the alphabet it is sent in: 306 characters, or 134 when it holds a character
outside the GSM alphabet. A failed delivery is tried again after one minute, then two, four and
so on up to an hour apart, twelve attempts over about six hours while the
finding stands, and is counted as failed, never dropped silently. A finding
that happens somewhere and arrives without its coordinates goes only to
subscribers who gave no place; it is not sent to every place.
Delivery works from the findings that stand, so someone who subscribes during
a flood is told of it. A condition (a flood, a storm) stands until a read no
longer shows it, and is told again when it comes back, unless it comes back
soon enough to be the same episode. How soon is the feed's to say, from its
own records (`research/watch_thresholds`): 26 hours for river gauges, most of
whose repeat floods are the next high tide (in thirty days of readings from
1,424 gauges, no gauge that dropped below flood stage returned between 26 and
30 hours, and 884 runs above flood stage are 138 episodes); 7.5 hours for
geomagnetic storms, which covers Kp dipping under G1 for one or two of its
three-hour intervals; two reads for a feed that has measured nothing. A
finding that worsens is told whatever the window. A single occurrence (an
earthquake) stands for a day: by then 85% of earthquakes have nine in ten of
the felt reports they will get. Subscribe only an address whose owner asked
for it.

## One watch per store

Any process that opens the store may run a watch: each worker of
`mercury-agent serve --workers 4` does when `MERCURY_WATCH_ENABLED=true`, and
so does `mercury-agent watch` beside a server. One of them holds the store's
lease for telling people and does the sending; each feed has its own lease
and is read by whichever watch holds it, so a `mercury-agent watch --feed
earthquakes` beside a server leaves the server reading the rest. A watch that
has no way of telling anyone does not take the lease for telling. The one
that tells delivers what any of them read, within a tick of its being
recorded. A feed is due when no watch of the store has read it for its
cadence, so the workers do not take turns reading a slow feed. The others
tell their own open consoles what is found, and take over a lease when its
holder stops (it gives them up at once on shutdown, and a delivery pass in
progress ends there) or 15 minutes after it last renewed. The store is one SQLite
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
- It places no calls. Text messages need the operator's own provider account,
  and no text was delivered in testing: the request was checked against the
  provider's live API only as far as authentication.
- An earthquake goes to the places where USGS says it was felt: where its
  ShakeMap estimates intensity III (weak shaking) or more, or where people in
  the place's 10 km cell reported it to "Did You Feel It?". The message says
  which. USGS publishes both some minutes to hours after an earthquake, and
  maps only the area around it: until it has published, and for a place its
  map does not cover and nobody reported from, the subscriber's own radius
  decides, as before. Mercury estimates no shaking itself.
- It reads nine feeds. The other loaders are either not live (benchmark files,
  series that end in 2024), need a key, or have no rule yet that would say
  something true and useful.
- River gauges and volcano alert levels are United States sources. Outside
  the United States a subscriber is told of their national authority's alerts
  where it publishes them to the WMO centre (most of Europe, China, Russia,
  Australia, Canada, parts of South America, Africa and Asia), of GDACS's
  Orange and Red events, and of earthquakes, tsunami buoys and geomagnetic
  storms. A country whose authority does not publish there (Japan and India
  among them) has no warning feed here.
- The WMO centre's alerts are looked up for subscribed places only; no alert
  outside the United States is reported without a place.
- A feed that cannot be read is shown as down and tried again at its cadence.

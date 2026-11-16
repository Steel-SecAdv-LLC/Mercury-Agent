# DART fixtures

Recorded from NOAA's National Data Buoy Center (public domain), unmodified
apart from being cut to the rows the tests need.

- `21418t2011_tohoku.txt`: station 21418, 2011-03-09 05:00 to 2011-03-13
  12:59 UTC, from `https://www.ndbc.noaa.gov/data/historical/dart/21418t2011.txt.gz`.
  The span `TsunamiLoader` reads for the `tohoku_2011` event, including the
  buoy's 1-minute and 15-second event-mode samples and the `9999.0` rows
  NDBC writes for samples that were not received.
- `46402_realtime_head.dart`: the first 40 lines of
  `https://www.ndbc.noaa.gov/data/realtime2/46402.dart` on 2026-11-16
  (newest first, as NDBC serves it).

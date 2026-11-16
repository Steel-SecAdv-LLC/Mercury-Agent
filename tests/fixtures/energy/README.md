# Energy fixtures

Recorded from U.S. government archives (public domain), unmodified apart
from being cut to the rows the tests need.

- `omni2_1989_quebec.dat`: NASA/GSFC OMNI2 hourly records for 1989-03-11 to
  1989-03-15, from `https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_1989.dat`.
  The March 1989 geomagnetic storm (Kp 9), including the hours with no
  solar-wind coverage (the archive's fill values).
- `omni2_2021_uri.dat`: the same archive for 2021-02-13 to 2021-02-19.
- `eia930_balance_2021_uri.csv`: EIA-930 Hourly Electric Grid Monitor rows
  from `https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2021_Jan_Jun.csv`:
  every ERCOT (`ERCO`) hour of 2021-02-13 to 2021-02-19, Winter Storm Uri,
  plus one day of another balancing authority (`CISO`) so that the reader's
  filter is exercised.
- `eia930_balance_2023_ice_storm.csv` and `omni2_2023_ice_storm.dat`: the same
  two archives for 2023-01-28 to 2023-02-08 (ERCOT only), the North Texas ice
  storm.
- `oe417_2023.json`: all 341 records of the DOE OE-417 2023 annual summary as
  served by Oak Ridge National Laboratory's Open Energy Hub
  (`https://openenergyhub.ornl.gov/api/explore/v2.1/catalog/datasets/oe-417-annual-summaries/records`),
  in one response-shaped document.
- `goes_xrs_1989_quebec.txt`: the rows for 1989-03-08 to 1989-03-19 of NOAA
  NCEI's GOES XRS flare report for 1989
  (`https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_1989.txt`).

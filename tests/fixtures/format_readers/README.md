# Recorded format samples (third-party files, byte-exact)

Consumed by `tests/loaders/test_format_readers_recorded.py`. Every file is
kept exactly as published (RINEX is fixed-column, so trailing blanks are
data); `.pre-commit-config.yaml` exempts this directory from the
whitespace and end-of-file fixers for that reason. Retrieved 2026-10-22.

| File | Source | Licence | SHA-256 |
|---|---|---|---|
| `demo.10o`, `minimal3.10o` | georinex test data, `src/georinex/tests/data/` (https://github.com/geospace-code/georinex) | MIT, `LICENSE-georinex.txt` | `b91455641e925e3a1884413fec3ccb83e0f99154815816d578b559f9af689013`, `ce3f8522e3d3d3e8857f0936e48fd136453509c55627edefb1db1a6bdb18b973` |
| `simple.las` | PDAL test data, `test/data/las/simple.las` (https://github.com/PDAL/PDAL) | BSD, `LICENSE-PDAL.txt` | `a0570ef57b685b77a6d3e3992cbdfeecdb2c3065d3780bbeaba490818258b734` |
| `gfs_20261021_t00z_f000_subset.grib2` | NOAA NCEP GFS 1.0-degree analysis, 2026-10-21 00Z f000, via NOMADS `filter_gfs_1p00.pl` with `var_TMP`, `var_PRMSL`, `lev_2_m_above_ground`, `lev_mean_sea_level`, subregion 40-50N 250-260E | US-government work, public domain | `012c81d5f62bd9539d3e042febc70fff3cd1dc2f70249ccba3d406e5b421395d` |
| `modbusSmall.pcap` | Zeek test traces, `testing/btest/Traces/modbus/modbusSmall.pcap` (https://github.com/zeek/zeek) | BSD, `COPYING-zeek.txt` | `329b4d75bb65d48102816b9d1b61015e7e03003bd38d70cc627522d9da2d46eb` |
| `zeek_modbus_coil_parsing_small.baseline` | Zeek's decoded baseline for that trace, `testing/btest/Baseline/scripts.base.protocols.modbus.coil_parsing_small/output` | BSD, `COPYING-zeek.txt` | `57b9647e6e9dc1cad3e8d037e635a4b99ce27cf09051c2fe8428f7a1c1e287c3` |
| `dnp3_read.pcap` | Zeek test traces, `testing/btest/Traces/dnp3/dnp3_read.pcap` | BSD, `COPYING-zeek.txt` | `1e39be441983fcf9acb4ee4a60ee986c4a9009bd8c1cc1f1de1b40e4e70fad65` |

Not vendored: the IEEE C37.118 captures from the Wireshark wiki
(https://wiki.wireshark.org/IEEE-C37.118) state no licence, so the test
downloads them under `@pytest.mark.network` and pins their SHA-256 instead.

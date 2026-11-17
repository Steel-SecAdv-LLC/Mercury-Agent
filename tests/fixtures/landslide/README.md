# Landslide fixtures

Recorded on 2026-11-17 from the public ArcGIS Online copy of NASA's Global
Landslide Catalog (layer `nasa_global_landslide_catalog_point`,
`https://services1.arcgis.com/yFGHRCyBneULM8ci/arcgis/rest/services/nasa_global_landslide_catalog_point/FeatureServer/0`;
14,750 rows, last edited 2025-08-10). Responses are as served, re-indented.

- `glc_mirror_oso_2014.json`: the answer to the query `LandslideLoader`
  makes for its `oso_2014` event (a box 200 km around 48.28, -121.84, from
  2014-03-01 to 2014-04-15): 24 reports, among them the 2014-03-22 Oso slide.
- `glc_mirror_selected_rows.json`: nine rows in one response-shaped
  document, taken from five queries: three of size `catastrophic`, two of
  category `rock_fall`, three dated before 2001, and one `Riverbank collapse`
  with no date.
- `glc_vocabulary.json`: every value of the category, trigger and size fields
  with its row count (three `groupByFieldsForStatistics` queries).

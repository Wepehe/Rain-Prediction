# Data-source audit

Verified 2026-08-31. “Verified” means the linked official documentation and at least one real object
or response were inspected. Availability statements distinguish operational rolling feeds from true
historical archives; those are not interchangeable.

## Recommended source stack

| Modality | Dataset and provider | Resolution / cadence | Verified history and access | Licence | Decision and caveats |
|---|---|---|---|---|---|
| Primary Canadian radar, live | ECCC North American radar composite, `RADAR_1KM_RRAI`, plus `RADAR_COVERAGE_RRAI` | 1 km; 6 min; rain rate in mm h⁻¹ | [MSC GeoMet WMS](https://eccc-msc.github.io/open-data/msc-data/obs_radar/readme_radar_geomet_en/) exposes only the last 3 hours | [ECCC data-server licence](https://eccc-msc.github.io/open-data/licence/readme_en/) | Correct live product and eventual label source. Save every issued cycle and coverage layer. It cannot bootstrap a multi-year archive by itself. |
| Canadian single-radar product | ECCC DPQPE | Native radar image; 6 min; rain mm h⁻¹ / snow cm h⁻¹ | [MSC Datamart](https://eccc-msc.github.io/open-data/msc-data/obs_radar/readme_radarimage-datamart_en/) rolling GIF feed; [metadata](https://open.canada.ca/data/en/dataset/6059da1d-e1da-4f2b-a420-b5c2a130eeaa) | Open Government Licence – Canada | Quality-controlled dual-pol estimate, but the public Datamart form is colour-mapped GIF rather than quantitative raster. S-band availability begins at different radar-upgrade dates. |
| Canadian radar, historical UI | ECCC Canadian Historical Weather Radar | 6 min images where available | [Climate archive UI](https://climate.weather.gc.ca/radar/index_e.html) | Government of Canada terms | Useful for case discovery and visual checks. The verified public interface is image-oriented; do not decode colours as the main quantitative training label without a validation study. A machine-readable quantitative national archive remains the main unresolved source issue. |
| Historical pilot radar | NOAA Multi-Radar/Multi-Sensor System (MRMS) `PrecipRate` | 0.01° (~1 km); 2 min; mm h⁻¹ | Anonymous `noaa-mrms-pds` S3 objects verified from 2020-10-14 onward; [official product table](https://www.nssl.noaa.gov/projects/mrms/operational/tables.php) | NOAA public data | Best reproducible pilot label for southern Ontario. Grid ends near 55°N and therefore cannot cover all Ontario. Preserve `-1` missing and `-3` no-coverage separately; the code maps both to NaN. |
| Border-radar supplement | NOAA NEXRAD Level II | Native polar gates; usually ~4–10 min volume scans | [NODD archive](https://registry.opendata.aws/noaa-nexrad/) from the early 1990s onward in `unidata-nexrad-level2` | NOAA public data | KBUF, KDTX, KTYX, KCLE and nearby sites can strengthen southern-border coverage and provide velocity/dual-pol fields. Geometry, blockage, mosaicking, and Canadian-domain range require explicit QC. |
| Satellite | NOAA GOES-East ABI calibrated imagery / CMI; cloud-top products | ABI IR nominally 2 km at nadir; CONUS commonly 5 min; derived cloud products commonly 10 min | [GOES NODD archive](https://registry.opendata.aws/noaa-goes/); GOES-16 historical mission data from 2017, with reprocessed L1b for early 2018–2024; operational GOES-East changed to GOES-19 in April 2025 | NOAA public data | Use calibrated channels (initially C08/C09/C10 water vapour and C13 IR) and derive 10–60 minute cooling tendencies. The pilot verified seven 2024 C13 CMI NetCDF files. Account for larger effective pixels and limb geometry over Ontario. |
| Satellite cloud top | NOAA ABI L2 cloud-top temperature/height/phase | ~2–10 km product-dependent; ~10 min | [NCEI cloud-top temperature record](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01507), prefix `ABI-L2-ACHT` | NOAA public data | Physically direct initiation features, but quality flags and algorithm-version changes must accompany the values. Compare C13 cooling against L2 cloud-top cooling rather than assuming either is superior. |
| Preferred Canadian NWP, live | ECCC HRDPS continental | 2.5 km; 00/06/12/18 UTC; hourly leads to 48 h; up to 31 pressure levels | [MSC Datamart GRIB2](https://eccc-msc.github.io/open-data/msc-data/nwp_hrdps/readme_hrdps-datamart_en/) | ECCC data-server licence | Correct live operational source. Datamart is a rolling “today” feed, not a verified multi-year archive. A stable public historical HRDPS archive has not yet been verified; contact ECCC and assess CaSPAr before scale-up. |
| Historical-pilot NWP | NOAA HRRR analysis/forecast | 3 km; hourly | [NODD archive since 2014](https://registry.opendata.aws/noaa-hrrr-pds/) | NOAA public data | Reproducible for southern Ontario only. The pilot byte-range downloader retrieves MSLP, 2 m T/Td, 10 m U/V and surface CAPE. HRRR assimilates radar, so “NWP adds independent pre-radar information” conclusions must control for assimilation and must eventually be repeated with HRDPS. |
| ECCC surface stations | Historical Climate Data hourly CSV; live MSC observations | Irregular stations; predominantly hourly historical, sub-hourly live at some sites | [historical search/download](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html); live [GeoMet retains 30 days](https://eccc-msc.github.io/open-data/msc-data/obs_station/readme_obs_insitu_en/) | Government of Canada terms / ECCC licence | Primary station source. The bulk CSV labels timestamps as local standard time; Toronto is interpreted as fixed UTC−05:00, not daylight time. Keep flags and station distances. The pilot verified Toronto Pearson and City Centre monthly files. |
| Station supplement | NOAA GHCN-H (successor to ISD) | Station-dependent, commonly hourly/sub-hourly | [NCEI documentation](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf) | NOAA public data | Cross-border and gap-filling source. Legacy ISD stopped updating after 2025-08-24, so new code should target GHCN-H rather than building on the superseded feed. |
| Lightning, optional | NOAA GOES GLM L2 | Event/group/flash records, ~20 s products | Same [GOES NODD archive](https://registry.opendata.aws/noaa-goes/) | NOAA public data | Open historical option from 2017. Ontario is far from the GLM sub-satellite point, so detection efficiency and geolocation uncertainty need validation. No unrestricted historical Canadian Lightning Detection Network archive was verified; do not block Milestone 1 on it. |

## Variables selected for the pilot

Radar uses MRMS instantaneous precipitation rate with explicit missingness. GOES starts with channel
13 brightness temperature because it is calibrated, physically interpretable, continuously available
at night, and supports cloud-top cooling. HRRR starts with 2 m temperature/dew point, 10 m wind,
mean-sea-level pressure, and surface CAPE. This is intentionally smaller than the desired production
set; pressure-level humidity/wind, CIN, vertical velocity, precipitable water, shear, convergence, and
lapse rates should be added only after their archive continuity and units pass an automated audit.

## Source-specific scientific risks

1. **Canadian quantitative radar history is unresolved.** The official 1 km composite is ideal live
   but has a three-hour GeoMet window. The official historical UI exposes imagery. NOAA MRMS is a
   scientifically useful southern-Ontario pilot, not an all-Ontario substitute.
2. **NWP issue-time leakage is easy to introduce.** Training examples must select only analyses and
   forecasts available at the simulated forecast issue time. Reanalysis or a later model cycle is not
   interchangeable with an operational input.
3. **HRRR is not an independent atmospheric-only control.** Its assimilation includes radar. Keep a
   plain NWP-precipitation baseline and repeat conclusions with HRDPS or another controlled source.
4. **Satellite geometry changes physical scale.** Store scan angle, projection, quality flags, source
   satellite, band, calibration, and actual pixel footprint; do not resize screenshots.
5. **Station time zones and moves matter.** Convert LST explicitly, retain climate/station IDs and
   quality flags, and use historical station metadata where possible.
6. **Product-version boundaries must become dataset strata.** Radar upgrades, MRMS revisions, GOES
   satellite transitions, and NWP upgrades can otherwise masquerade as meteorological skill.

## Retrieval code

- `ontario_nowcast.data.sample`: anonymous MRMS discovery, immutable download, GRIB2 decode, crop,
  and processed build manifest.
- `ontario_nowcast.data.multimodal`: GOES C13 object discovery, HRRR `.idx` byte-range subsetting,
  ECCC hourly CSV retrieval, and availability/time-offset synchronization table.
- `ontario_nowcast.preprocessing.fuse`: projection-aware radar/GOES interpolation, HRRR grid mapping,
  distance-limited station features, and a common 2 km EPSG:3978 tensor.
- `ontario_nowcast.data.manifest`: atomic no-overwrite downloads with retrieval timestamp, size,
  SHA-256, source timestamp, and product metadata.

Large data are ignored by Git. The checked-in configuration is the reproducible request description;
`data/metadata/*.jsonl` is the local build record.

# BreizhCrops FRH04 evidence sample

- Official archive: `https://breizhcrops.s3.eu-central-1.amazonaws.com/2017/L1C/frh04.zip`
- Dataset version: 2017, Sentinel-2 L1C (top-of-atmosphere), FRH04/Morbihan
- Archive size reported by the server: 1,547,576,548 bytes
- Archive inventory obtained from its ZIP64 central directory: 158,338 CSV files and one directory entry
- Extracted member: `frh04/6082638.csv`
- Member size: 32,218 bytes uncompressed; 12,546 bytes compressed
- Member CRC-32: 561438952 (verified against the ZIP entry)
- Member SHA-256: `8d5c29958da23234ec09b72fbfb3520bc2680cc57b39c29c85c83fde41e00832`
- Observations: 102 rows, 102 unique dates, 2017-01-02 through 2017-12-28
- Field ID: 6082638
- Raw crop code: ORH
- Official class mapping: class ID 0, `barley`

Raw CSV columns:

`[unnamed acquisition-id index], B1, B10, B11, B12, B2, B3, B4, B5, B6, B7, B8, B8A, B9, QA10, QA20, QA60, doa, label, id`

The official loader first builds a 102 x 17 array for this sample in the order
`B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12, QA10, QA20, QA60, doa`.
Its default L1C transform then keeps the 13 spectral bands, multiplies them by
`1e-4`, and randomly selects 45 time steps without replacement, producing a
45 x 13 model input. The target is scalar class ID 0.

# Stage 3 -- Subband-wise Master Comparison

*60/72 (candidate, disease, subband) rows evaluated. Complements Stage3_Disease_Master_Comparison.md (overall similarity); this table shows which frequency band -- A3 (slow-wave/morphology), D3, D2, D1 (high-frequency QRS) -- diverges most from real ECGs, per mentor_eval/subband_features.py's bior4.4/level-3 decomposition. No Hausdorff column (not computed per-band). AFIB is not a row here -- it has no entry in subband_similarity_metrics.py's own class list, not filtered out by this script.*

| Candidate | Disease | Subband | Cosine | Mahalanobis | Bhattacharyya | Status |
|---|---|---|---|---|---|---|
| S3-001 | Normal | A3 | 0.1379 | 4.3418 | 8.3548 | done |
| S3-001 | Normal | D3 | 0.1379 | 2.1521 | 4.5346 | done |
| S3-001 | Normal | D2 | 0.1379 | 1.7584 | 4.5473 | done |
| S3-001 | Normal | D1 | 0.1379 | 1.2008 | 1.9471 | done |
| S3-001 | STEMI | A3 | 0.1225 | 5.9645 | 8.5623 | done |
| S3-001 | STEMI | D3 | 0.1225 | 2.0336 | 6.2401 | done |
| S3-001 | STEMI | D2 | 0.1225 | 1.3615 | 5.7709 | done |
| S3-001 | STEMI | D1 | 0.1225 | 1.0758 | 2.1514 | done |
| S3-001 | NSTEMI | A3 | 0.1201 | 5.7633 | 10.3445 | done |
| S3-001 | NSTEMI | D3 | 0.1201 | 2.034 | 7.4827 | done |
| S3-001 | NSTEMI | D2 | 0.1201 | 1.4055 | 6.9633 | done |
| S3-001 | NSTEMI | D1 | 0.1201 | 1.0722 | 3.1784 | done |
| S3-002 | Normal | A3 | 0.1396 | 5.9302 | 7.1605 | done |
| S3-002 | Normal | D3 | 0.1396 | 2.355 | 4.2272 | done |
| S3-002 | Normal | D2 | 0.1396 | 1.7925 | 4.6278 | done |
| S3-002 | Normal | D1 | 0.1396 | 1.2043 | 1.9551 | done |
| S3-002 | STEMI | A3 | 0.1212 | 8.7355 | 7.3275 | done |
| S3-002 | STEMI | D3 | 0.1212 | 2.299 | 5.6567 | done |
| S3-002 | STEMI | D2 | 0.1212 | 1.3911 | 5.6538 | done |
| S3-002 | STEMI | D1 | 0.1212 | 1.1163 | 2.0646 | done |
| S3-002 | NSTEMI | A3 | 0.1253 | 6.9282 | 9.9412 | done |
| S3-002 | NSTEMI | D3 | 0.1253 | 2.3326 | 7.8786 | done |
| S3-002 | NSTEMI | D2 | 0.1253 | 1.4979 | 7.6143 | done |
| S3-002 | NSTEMI | D1 | 0.1253 | 1.119 | 3.3163 | done |
| S3-003 | Normal | A3 | 0.1397 | 6.0466 | 7.7784 | done |
| S3-003 | Normal | D3 | 0.1397 | 2.4822 | 4.0134 | done |
| S3-003 | Normal | D2 | 0.1397 | 1.8134 | 4.058 | done |
| S3-003 | Normal | D1 | 0.1397 | 1.1741 | 1.8375 | done |
| S3-003 | STEMI | A3 | 0.1276 | 7.4341 | 8.1953 | done |
| S3-003 | STEMI | D3 | 0.1276 | 2.2383 | 6.0943 | done |
| S3-003 | STEMI | D2 | 0.1276 | 1.4131 | 5.6411 | done |
| S3-003 | STEMI | D1 | 0.1276 | 1.1621 | 2.1568 | done |
| S3-003 | NSTEMI | A3 | 0.1255 | 7.3105 | 10.2001 | done |
| S3-003 | NSTEMI | D3 | 0.1255 | 2.3296 | 7.5675 | done |
| S3-003 | NSTEMI | D2 | 0.1255 | 1.4892 | 6.9236 | done |
| S3-003 | NSTEMI | D1 | 0.1255 | 1.1008 | 3.1653 | done |
| S3-004 | Normal | A3 | 0.1391 | 4.9173 | 7.1274 | done |
| S3-004 | Normal | D3 | 0.1391 | 2.1179 | 3.9092 | done |
| S3-004 | Normal | D2 | 0.1391 | 1.7435 | 4.256 | done |
| S3-004 | Normal | D1 | 0.1391 | 1.205 | 1.9207 | done |
| S3-004 | STEMI | A3 | 0.1284 | 6.2029 | 7.5701 | done |
| S3-004 | STEMI | D3 | 0.1284 | 2.0205 | 5.8115 | done |
| S3-004 | STEMI | D2 | 0.1284 | 1.3968 | 5.5928 | done |
| S3-004 | STEMI | D1 | 0.1284 | 1.1264 | 2.1497 | done |
| S3-004 | NSTEMI | A3 | 0.1229 | 6.0937 | 9.5919 | done |
| S3-004 | NSTEMI | D3 | 0.1229 | 2.145 | 7.2258 | done |
| S3-004 | NSTEMI | D2 | 0.1229 | 1.4426 | 7.0719 | done |
| S3-004 | NSTEMI | D1 | 0.1229 | 1.1084 | 3.3082 | done |
| S3-005 | Normal | A3 | 0.1367 | 5.6793 | 7.2991 | done |
| S3-005 | Normal | D3 | 0.1367 | 2.6769 | 3.4144 | done |
| S3-005 | Normal | D2 | 0.1367 | 1.8026 | 3.9712 | done |
| S3-005 | Normal | D1 | 0.1367 | 1.1877 | 1.7183 | done |
| S3-005 | STEMI | A3 | 0.1224 | 8.2715 | 7.429 | done |
| S3-005 | STEMI | D3 | 0.1224 | 2.4462 | 5.1726 | done |
| S3-005 | STEMI | D2 | 0.1224 | 1.3915 | 5.2509 | done |
| S3-005 | STEMI | D1 | 0.1224 | 1.1085 | 1.8526 | done |
| S3-005 | NSTEMI | A3 | 0.1245 | 5.8508 | 10.6256 | done |
| S3-005 | NSTEMI | D3 | 0.1245 | 2.4257 | 7.5887 | done |
| S3-005 | NSTEMI | D2 | 0.1245 | 1.5306 | 7.34 | done |
| S3-005 | NSTEMI | D1 | 0.1245 | 1.1225 | 3.2091 | done |
| S3-006 | Normal | A3 | -- | -- | -- | not yet evaluated -- no subband_similarity_metrics.csv found |
| S3-006 | Normal | D3 | -- | -- | -- | not yet evaluated -- no subband_similarity_metrics.csv found |
| S3-006 | Normal | D2 | -- | -- | -- | not yet evaluated -- no subband_similarity_metrics.csv found |
| S3-006 | Normal | D1 | -- | -- | -- | not yet evaluated -- no subband_similarity_metrics.csv found |
| S3-006 | STEMI | A3 | -- | -- | -- | not yet evaluated -- no subband_similarity_metrics.csv found |
| S3-006 | STEMI | D3 | -- | -- | -- | not yet evaluated -- no subband_similarity_metrics.csv found |
| S3-006 | STEMI | D2 | -- | -- | -- | not yet evaluated -- no subband_similarity_metrics.csv found |
| S3-006 | STEMI | D1 | -- | -- | -- | not yet evaluated -- no subband_similarity_metrics.csv found |
| S3-006 | NSTEMI | A3 | -- | -- | -- | not yet evaluated -- no subband_similarity_metrics.csv found |
| S3-006 | NSTEMI | D3 | -- | -- | -- | not yet evaluated -- no subband_similarity_metrics.csv found |
| S3-006 | NSTEMI | D2 | -- | -- | -- | not yet evaluated -- no subband_similarity_metrics.csv found |
| S3-006 | NSTEMI | D1 | -- | -- | -- | not yet evaluated -- no subband_similarity_metrics.csv found |

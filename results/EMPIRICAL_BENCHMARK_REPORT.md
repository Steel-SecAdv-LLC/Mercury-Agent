# OMNI-AVA Empirical Benchmark Report

**Generated:** 2025-12-08T02:36:45.495210+00:00

## Methodology

This benchmark compares OMNI-AVA against established anomaly detection algorithms using publicly available datasets from scikit-learn.

### Datasets

- breast_cancer
- digits_8
- covtype
- kddcup99

### Baseline Detectors

- IsolationForest
- OneClassSVM
- LocalOutlierFactor
- EllipticEnvelope

## Results Summary

| Detector | Mean ROC-AUC | Mean F1 | Mean Latency (ms) |
|----------|--------------|---------|-------------------|
| EllipticEnvelope | 0.763 | 0.244 | 0.002 |
| IsolationForest | 0.756 | 0.269 | 0.015 |
| OneClassSVM | 0.700 | 0.190 | 0.009 |
| LocalOutlierFactor | 0.593 | 0.223 | 0.009 |
| OMNI-AVA | 0.201 | 0.247 | 0.268 |

## Honest Assessment

**Verdict:** OMNI-AVA ranks #5, 0.562 ROC-AUC below best baseline

### Methodology Notes

- Benchmarks use publicly available sklearn datasets
- Anomaly labels derived from minority class designation
- All detectors use same train/test splits for fair comparison
- Contamination parameter set based on actual anomaly ratio

### Limitations

- Datasets are proxies for real-world anomaly detection scenarios
- Medical dataset (breast_cancer) is not actual clinical data
- Cybersecurity dataset (KDDCup99) is from 1999, may not reflect modern attacks
- Results may vary with different random seeds and hyperparameters

**Recommendation:** For production use, validate on domain-specific real-world data. These benchmarks provide directional guidance only.

## Detailed Results

### OMNI-AVA on breast_cancer

- ROC-AUC: 0.2284
- PR-AUC: -0.2490
- F1 Score: 0.5938
- Precision: 0.5938
- Recall: 0.5938
- False Positive Rate: 0.2430
- Inference Latency: 0.7129 ms/sample

### IsolationForest on breast_cancer

- ROC-AUC: 0.8194
- PR-AUC: -0.6851
- F1 Score: 0.6777
- Precision: 0.7193
- Recall: 0.6406
- False Positive Rate: 0.1495
- Inference Latency: 0.0347 ms/sample

### OneClassSVM on breast_cancer

- ROC-AUC: 0.6703
- PR-AUC: -0.5473
- F1 Score: 0.5397
- Precision: 0.5484
- Recall: 0.5312
- False Positive Rate: 0.2617
- Inference Latency: 0.0094 ms/sample

### LocalOutlierFactor on breast_cancer

- ROC-AUC: 0.6060
- PR-AUC: -0.4308
- F1 Score: 0.4298
- Precision: 0.4561
- Recall: 0.4062
- False Positive Rate: 0.2897
- Inference Latency: 0.0117 ms/sample

### EllipticEnvelope on breast_cancer

- ROC-AUC: 0.8493
- PR-AUC: -0.7162
- F1 Score: 0.7059
- Precision: 0.6667
- Recall: 0.7500
- False Positive Rate: 0.2243
- Inference Latency: 0.0027 ms/sample

### OMNI-AVA on digits_8

- ROC-AUC: 0.4418
- PR-AUC: -0.0803
- F1 Score: 0.0381
- Precision: 0.0377
- Recall: 0.0385
- False Positive Rate: 0.1045
- Inference Latency: 0.2677 ms/sample

### IsolationForest on digits_8

- ROC-AUC: 0.3500
- PR-AUC: -0.0680
- F1 Score: 0.0182
- Precision: 0.0172
- Recall: 0.0192
- False Positive Rate: 0.1168
- Inference Latency: 0.0154 ms/sample

### OneClassSVM on digits_8

- ROC-AUC: 0.2335
- PR-AUC: -0.0580
- F1 Score: 0.0169
- Precision: 0.0152
- Recall: 0.0192
- False Positive Rate: 0.1332
- Inference Latency: 0.0074 ms/sample

### LocalOutlierFactor on digits_8

- ROC-AUC: 0.5652
- PR-AUC: -0.1037
- F1 Score: 0.0588
- Precision: 0.0600
- Recall: 0.0577
- False Positive Rate: 0.0963
- Inference Latency: 0.0070 ms/sample

### EllipticEnvelope on digits_8

- ROC-AUC: 0.4968
- PR-AUC: -0.0873
- F1 Score: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- False Positive Rate: 0.0984
- Inference Latency: 0.0022 ms/sample

### OMNI-AVA on covtype

- ROC-AUC: 0.0630
- PR-AUC: -0.0025
- F1 Score: 0.1000
- Precision: 0.0741
- Recall: 0.1538
- False Positive Rate: 0.0093
- Inference Latency: 0.0452 ms/sample

### IsolationForest on covtype

- ROC-AUC: 0.9159
- PR-AUC: -0.0426
- F1 Score: 0.0952
- Precision: 0.0690
- Recall: 0.1538
- False Positive Rate: 0.0100
- Inference Latency: 0.0053 ms/sample

### OneClassSVM on covtype

- ROC-AUC: 0.9245
- PR-AUC: -0.0475
- F1 Score: 0.0870
- Precision: 0.0536
- Recall: 0.2308
- False Positive Rate: 0.0197
- Inference Latency: 0.0095 ms/sample

### LocalOutlierFactor on covtype

- ROC-AUC: 0.6506
- PR-AUC: -0.0243
- F1 Score: 0.1000
- Precision: 0.0741
- Recall: 0.1538
- False Positive Rate: 0.0093
- Inference Latency: 0.0096 ms/sample

### EllipticEnvelope on covtype

- ROC-AUC: 0.9162
- PR-AUC: -0.1256
- F1 Score: 0.2703
- Precision: 0.2083
- Recall: 0.3846
- False Positive Rate: 0.0071
- Inference Latency: 0.0010 ms/sample

### OMNI-AVA on kddcup99

- ROC-AUC: 0.0714
- PR-AUC: -0.0162
- F1 Score: 0.2543
- Precision: 0.2418
- Recall: 0.2683
- False Positive Rate: 0.0264
- Inference Latency: 0.0453 ms/sample

### IsolationForest on kddcup99

- ROC-AUC: 0.9372
- PR-AUC: -0.4058
- F1 Score: 0.2840
- Precision: 0.2875
- Recall: 0.2805
- False Positive Rate: 0.0218
- Inference Latency: 0.0050 ms/sample

### OneClassSVM on kddcup99

- ROC-AUC: 0.9731
- PR-AUC: -0.3330
- F1 Score: 0.1154
- Precision: 0.1216
- Recall: 0.1098
- False Positive Rate: 0.0248
- Inference Latency: 0.0116 ms/sample

### LocalOutlierFactor on kddcup99

- ROC-AUC: 0.5497
- PR-AUC: -0.3092
- F1 Score: 0.3038
- Precision: 0.3158
- Recall: 0.2927
- False Positive Rate: 0.0199
- Inference Latency: 0.0089 ms/sample

### EllipticEnvelope on kddcup99

- ROC-AUC: 0.7890
- PR-AUC: -0.0635
- F1 Score: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- False Positive Rate: 0.0378
- Inference Latency: 0.0006 ms/sample


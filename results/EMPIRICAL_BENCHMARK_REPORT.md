# Mercury-Agent Empirical Benchmark Report

**Generated:** 2026-10-28T03:26:13.739914+00:00

## Methodology

This benchmark compares Mercury-Agent against established anomaly detection algorithms using publicly available datasets from scikit-learn.

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
| IsolationForest | 0.829 | 0.273 | 0.029 |
| OneClassSVM | 0.824 | 0.262 | 0.014 |
| EllipticEnvelope | 0.785 | 0.266 | 0.002 |
| Mercury-Agent | 0.736 | 0.130 | 0.041 |
| LocalOutlierFactor | 0.588 | 0.184 | 0.014 |

## Honest Assessment

**Verdict:** Mercury-Agent ranks #4, 0.093 ROC-AUC below best baseline

### Methodology Notes

- Benchmarks use publicly available sklearn datasets
- Anomaly labels derived from minority class designation
- All detectors use same train/test splits for fair comparison
- Contamination parameter set based on actual anomaly ratio

### Limitations

- Datasets are proxies for real-world anomaly detection scenarios
- Medical dataset (breast_cancer) is not actual clinical data
- Cybersecurity dataset (KDDCup99) is from 1999, may not reflect modern attacks
- A run is deterministic for a fixed --seed and dataset snapshot; absolute scores can shift across NumPy/scikit-learn versions and hyperparameters

**Recommendation:** For production use, validate on domain-specific real-world data. These benchmarks provide directional guidance only.

## Detailed Results

### Mercury-Agent on breast_cancer

- ROC-AUC: 0.7897
- PR-AUC: 0.6336
- F1 Score: 0.3856
- Precision: 0.6701
- Recall: 0.2831
- False Positive Rate: 0.0869
- Inference Latency: 0.0267 ms/sample

### IsolationForest on breast_cancer

- ROC-AUC: 0.7800
- PR-AUC: 0.6429
- F1 Score: 0.6397
- Precision: 0.6255
- Recall: 0.6557
- False Positive Rate: 0.2325
- Inference Latency: 0.0729 ms/sample

### OneClassSVM on breast_cancer

- ROC-AUC: 0.6620
- PR-AUC: 0.5442
- F1 Score: 0.5253
- Precision: 0.5104
- Recall: 0.5426
- False Positive Rate: 0.3108
- Inference Latency: 0.0185 ms/sample

### LocalOutlierFactor on breast_cancer

- ROC-AUC: 0.5439
- PR-AUC: 0.4315
- F1 Score: 0.4075
- Precision: 0.4004
- Recall: 0.4150
- False Positive Rate: 0.3697
- Inference Latency: 0.0151 ms/sample

### EllipticEnvelope on breast_cancer

- ROC-AUC: 0.8880
- PR-AUC: 0.7939
- F1 Score: 0.7534
- Precision: 0.7622
- Recall: 0.7502
- False Positive Rate: 0.1430
- Inference Latency: 0.0053 ms/sample

### Mercury-Agent on digits_8

- ROC-AUC: 0.5507
- PR-AUC: 0.1105
- F1 Score: 0.0370
- Precision: 0.0636
- Recall: 0.0289
- False Positive Rate: 0.0493
- Inference Latency: 0.0515 ms/sample

### IsolationForest on digits_8

- ROC-AUC: 0.6533
- PR-AUC: 0.1867
- F1 Score: 0.0284
- Precision: 0.0281
- Recall: 0.0289
- False Positive Rate: 0.1023
- Inference Latency: 0.0271 ms/sample

### OneClassSVM on digits_8

- ROC-AUC: 0.7673
- PR-AUC: 0.3209
- F1 Score: 0.0262
- Precision: 0.0241
- Recall: 0.0287
- False Positive Rate: 0.1282
- Inference Latency: 0.0111 ms/sample

### LocalOutlierFactor on digits_8

- ROC-AUC: 0.5710
- PR-AUC: 0.1156
- F1 Score: 0.0886
- Precision: 0.0862
- Recall: 0.0921
- False Positive Rate: 0.0980
- Inference Latency: 0.0086 ms/sample

### EllipticEnvelope on digits_8

- ROC-AUC: 0.5025
- PR-AUC: 0.0941
- F1 Score: 0.0061
- Precision: 0.0065
- Recall: 0.0057
- False Positive Rate: 0.1140
- Inference Latency: 0.0028 ms/sample

### Mercury-Agent on covtype

- ROC-AUC: 0.8986
- PR-AUC: 0.0379
- F1 Score: 0.0515
- Precision: 0.0280
- Recall: 0.3583
- False Positive Rate: 0.0594
- Inference Latency: 0.0595 ms/sample

### IsolationForest on covtype

- ROC-AUC: 0.9364
- PR-AUC: 0.0979
- F1 Score: 0.1184
- Precision: 0.0868
- Recall: 0.1889
- False Positive Rate: 0.0094
- Inference Latency: 0.0073 ms/sample

### OneClassSVM on covtype

- ROC-AUC: 0.9012
- PR-AUC: 0.0494
- F1 Score: 0.0828
- Precision: 0.0493
- Recall: 0.2611
- False Positive Rate: 0.0244
- Inference Latency: 0.0100 ms/sample

### LocalOutlierFactor on covtype

- ROC-AUC: 0.6672
- PR-AUC: 0.0248
- F1 Score: 0.0487
- Precision: 0.0395
- Recall: 0.0694
- False Positive Rate: 0.0084
- Inference Latency: 0.0151 ms/sample

### EllipticEnvelope on covtype

- ROC-AUC: 0.8986
- PR-AUC: 0.0843
- F1 Score: 0.1127
- Precision: 0.0826
- Recall: 0.1806
- False Positive Rate: 0.0094
- Inference Latency: 0.0012 ms/sample

### Mercury-Agent on kddcup99

- ROC-AUC: 0.7059
- PR-AUC: 0.0583
- F1 Score: 0.0476
- Precision: 0.0355
- Recall: 0.0749
- False Positive Rate: 0.0704
- Inference Latency: 0.0245 ms/sample

### IsolationForest on kddcup99

- ROC-AUC: 0.9470
- PR-AUC: 0.4679
- F1 Score: 0.3059
- Precision: 0.2992
- Recall: 0.3140
- False Positive Rate: 0.0249
- Inference Latency: 0.0080 ms/sample

### OneClassSVM on kddcup99

- ROC-AUC: 0.9662
- PR-AUC: 0.3088
- F1 Score: 0.4131
- Precision: 0.3395
- Recall: 0.5312
- False Positive Rate: 0.0317
- Inference Latency: 0.0165 ms/sample

### LocalOutlierFactor on kddcup99

- ROC-AUC: 0.5709
- PR-AUC: 0.0609
- F1 Score: 0.1921
- Precision: 0.1936
- Recall: 0.1912
- False Positive Rate: 0.0270
- Inference Latency: 0.0164 ms/sample

### EllipticEnvelope on kddcup99

- ROC-AUC: 0.8512
- PR-AUC: 0.2247
- F1 Score: 0.1933
- Precision: 0.1953
- Recall: 0.1920
- False Positive Rate: 0.0279
- Inference Latency: 0.0006 ms/sample


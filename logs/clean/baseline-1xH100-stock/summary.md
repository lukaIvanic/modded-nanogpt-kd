# Baseline Run Summary

**Purpose**: Validate stock train_gpt.py on single H100 instance

## Results

| Metric | Value |
|--------|-------|
| Final Val Loss | 3.2787 |
| Total Steps | 1530 |
| Total Time | 673.4s (~11.2 min) |
| Avg Step Time | 440.12ms |
| Peak Memory | 37,326 MiB allocated |

## Validation Loss Progression

| Step | Val Loss |
|------|----------|
| 0 | 10.8295 |
| 250 | 4.5314 |
| 500 | 4.3692 |
| 750 | 3.8644 |
| 1000 | 3.5587 |
| 1250 | 3.3946 |
| 1500 | 3.2884 |
| 1530 | 3.2787 |

## Notes

- Healthy downward loss curve confirmed
- Final val_loss 3.2787 very close to 8xH100 target of <=3.28
- Instance validated and ready for KD experiments

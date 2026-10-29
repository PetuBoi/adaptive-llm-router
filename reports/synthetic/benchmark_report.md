# RouteLab benchmark

Origin: **SIMULATED**. Dataset: demo.

Macro mean of task means on published [0,1] scores; utility proxy, not universal accuracy

| Strategy | Macro task quality | Mean reference USD | Mean latency ms | Failure rate |
|---|---:|---:|---:|---:|
| cheapest | 0.6431999569034227 | 3.28e-05 | 427.56610474658567 | 0.0 |
| strongest | 0.8504258373256846 | 0.0009674404761904761 | 1263.2954868430952 | 0.0 |
| random | 0.7579838325244643 | 0.00042400714285714274 | 870.9487735083547 | 0.0 |
| rule | 0.6995369195367559 | 0.0003851761904761904 | 730.2513278949642 | 0.0 |
| ml | 0.8290744105420536 | 0.0006082071428571428 | 967.7041569610215 | 0.0 |
| oracle | 0.8554251771641667 | 0.000808190476190476 | 1129.2010971229572 | 0.0 |

Common successful cohort: 42 prompts.
Task-level definitions and scores: task_performance.csv.

Reference prices and recorded latency are historical. Metrics differ by task. No significance claim.

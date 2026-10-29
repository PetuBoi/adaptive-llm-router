# RouteLab benchmark

Origin: **RECORDED**. Dataset: xroutebench.

Macro mean of task means on published [0,1] scores; utility proxy, not universal accuracy

| Strategy | Macro task quality | Mean reference USD | Mean latency ms | Failure rate |
|---|---:|---:|---:|---:|
| cheapest | 0.581591428753116 | 1.1937382676320729e-05 | 3170.9050683829446 | 0.0 |
| strongest | 0.6535553303348726 | 0.0004576803432555645 | 2521.0632877447038 | 0.0 |
| random | 0.5284102686026044 | 0.0001816433494234379 | 2867.78653794583 | 0.0 |
| rule | 0.6292291565086988 | 0.000285819066773934 | 2712.826226870475 | 0.0 |
| ml | 0.6849642234137691 | 8.599231965674443e-05 | 3198.183159023867 | 0.0 |
| oracle | 0.8532747553548466 | 3.234866988468758e-05 | 2636.6173236792706 | 0.0 |

Common successful cohort: 3729 prompts.
Task-level definitions and scores: task_performance.csv.

Reference prices and recorded latency are historical. Metrics differ by task. No significance claim.

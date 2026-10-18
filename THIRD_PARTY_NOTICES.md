# Data, dependencies, and references

## Loghub / Blue Gene/L

Source: https://github.com/logpai/loghub/tree/master/BGL

Archive: https://zenodo.org/records/8196385/files/BGL.zip?download=1

Loghub states that its datasets are freely available for research or academic work, requests a reference to its repository for usage/distribution, and asks for citation of the Loghub paper where applicable. Consult the current [Loghub license and terms](https://github.com/logpai/loghub#-license) before another use. Project code's MIT license does not override those terms.

`artifacts/portfolio/traces.json` and the vocabulary contain normalized content derived from the BGL dataset, with labels and source line indexes. Other artifacts include learned model parameters and statistical summaries. The original raw archive is not bundled. Preserve this notice when sharing the research project.

Required BGL/Loghub citations:

- Adam J. Oliner and Jon Stearley. **What Supercomputers Say: A Study of Five System Logs.** IEEE/IFIP International Conference on Dependable Systems and Networks (DSN), 2007.
- Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, and Michael R. Lyu. **Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics.** IEEE International Symposium on Software Reliability Engineering (ISSRE), 2023. https://arxiv.org/abs/2008.06448

Loghub also requests citation of its related parsing evaluation:

- Zhihan Jiang et al. **A Large-scale Evaluation for Log Parsing Techniques: How Far are We?** ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA), 2024. https://arxiv.org/abs/2308.10828

## Implementation references

- PyTorch LSTM API: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- PyTorch state dictionaries and serialization: https://docs.pytorch.org/docs/stable/notes/serialization.html

TraceLens is an independent implementation. It is not affiliated with WGU, Loghub, LLNL, or the PyTorch maintainers. The D802 mapping is based on the course overview supplied for this project, not an official assessment rubric.

Dependencies retain their respective licenses. `requirements-lock.txt` records the verified installed versions. No external image or font assets are needed by the application.

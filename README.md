# FuzzyCS: Adaptive Client Selection via Fuzzy Rough Approximations in Federated Learning

## Contributions
- **Fuzzy-Rough-Set-Based Client Selection Framework**: We propose a novel fuzzy rough set perspective for federated client selection. A criteria-aware fuzzy concept is defined to build the multi-criteria information matrix, offering a formal formulation to capture uncertain and imprecise information during client selection.
- **Generalized Fuzzy Rough Selection Model**: Generalized fuzzy rough lower and upper approximations are adopted to quantify how well each client meets multiple criteria. On this basis, FuzzyCS is developed, converting ad-hoc selection heuristics into rigorous fuzzy membership estimation.
- **Trade-off Modeling Free of Auxiliary Optimization**: FuzzyCS aggregates statistical and system-level criteria via fuzzy rough scores. Such scores directly produce client sampling weights, removing the need for extra numerical optimization required by many hybrid competitors.
- **Comprehensive Empirical Validation**: We conduct experiments on image, text and sensor benchmark datasets against 11 baselines. The Friedman test confirms that FuzzyCS achieves the best average rank over all metrics and datasets, demonstrating its superior overall trade-off performance.

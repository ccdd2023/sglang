# Cross-role tail spans (d_norm > 0.5), by (granularity, path)

Source: ast_granularity_distance_7b.json, cross-role cells only (planner excluded).

Threshold: d_norm > 0.5.

Sorted by max cross-role d_norm, descending.


| Granularity | Path | n cells | tail cells | tail rate | max d_norm | max role | span tokens | lines | repo |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| class | lib/matplotlib/patches.py | 30 | 6 | 20.0% | 0.770 | coder | 244 | 3560-3591 |  |
| statement_window | lib/matplotlib/axes/_base.py | 4 | 2 | 50.0% | 0.750 | coder | 137 | 521-532 |  |
| class | pylint/checkers/base.py | 8 | 5 | 62.5% | 0.658 | coder | 265 | 84-111 |  |
| statement_window | testing/python/metafunc.py | 4 | 4 | 100.0% | 0.650 | coder | 154 | 161-172 |  |
| statement_window | lib/matplotlib/patches.py | 4 | 2 | 50.0% | 0.627 | coder | 220 | 2281-2292 |  |
| control_block | lib/matplotlib/axes/_base.py | 8 | 2 | 25.0% | 0.585 | coder | 174 | 1694-1710 |  |
| method | sklearn/linear_model/coordinate_descent.py | 2 | 2 | 100.0% | 0.576 | coder | 215 | 1564-1584 |  |
| method | pylint/checkers/base.py | 10 | 2 | 20.0% | 0.574 | coder | 223 | 1849-1875 |  |
| file_prefix | src/flask/scaffold.py | 2 | 2 | 100.0% | 0.573 | coder | 2214 | 1-240 |  |
| file_prefix | pylint/checkers/base.py | 2 | 2 | 100.0% | 0.546 | coder | 1994 | 1-120 |  |
| control_block | seaborn/axisgrid.py | 2 | 1 | 50.0% | 0.546 | coder | 176 | 2317-2334 |  |
| control_block | lib/matplotlib/axes/_axes.py | 2 | 2 | 100.0% | 0.546 | coder | 172 | 2373-2394 |  |
| class | requests/models.py | 2 | 1 | 50.0% | 0.536 | coder | 134 | 136-154 |  |
| method | django/contrib/admin/options.py | 6 | 1 | 16.7% | 0.514 | coder | 224 | 1487-1503 |  |

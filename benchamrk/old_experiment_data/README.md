# Dataset

This directory contains the dataset.

> Important: these files are not the original experimental data used in the paper. They were produced after updating parts of the implementation and rerunning the full experiment. The tested dataset is the same.

# Mutation Generation

Mutation results were generated with `mutmut 3.4.0` using its LibCST-based node mutation rules.

| Category | Mutation rule |
| --- | --- |
| Numbers | `1 -> 2`, `1.5 -> 2.5`, `1j -> 2j` |
| Strings | `"x" -> "XXxXX"`, lower/upper; triple-quoted strings are skipped |
| Booleans and names | `True <-> False`, `deepcopy -> copy` |
| Assignments | `x = value -> x = None`, `x = None -> x = ""` |
| Augmented assignments | `x += y -> x = y`, `+= <-> -=`, `*= <-> /=`, and related operator swaps |
| Arithmetic and bit operators | `+ <-> -`, `* <-> /`, `// -> /`, `% -> /`, `** -> *`, `<< <-> >>`, `& <-> \|`, `^ -> &` |
| Comparisons and logic | `< <-> <=`, `> <-> >=`, `== <-> !=`, `and <-> or` |
| Membership, identity, unary | `in <-> not in`, `is <-> is not`, remove `not`/`~`, unary `+ <-> -` |
| Function calls | argument `-> None`; remove one argument from multi-argument calls |
| `dict(...)` calls | `dict(a=b) -> dict(aXX=b)` |
| String methods | `lower <-> upper`, `lstrip <-> rstrip`, `find <-> rfind`, `split <-> rsplit`, and related pairs |
| Lambdas | `lambda ...: expr -> lambda ...: None`, `lambda ...: None -> lambda ...: 0` |
| Control flow and match | `break -> return`, `continue -> break`, remove one `case` from multi-case `match` |

# Performance

| Scope | PROBE mutation score | Baseline mutation score |
| --- | ---: | ---: |
| Overall | 80.15% | 71.61% |
| Passing intersection | 82.80% | 72.14% |

# Package-Level Performance

| Package | PROBE mutation score | Baseline mutation score |
| --- | ---: | ---: |
| sympy | 79.36% | 65.10% |
| sortedcontainers | 89.15% | 79.97% |
| more-itertools | 84.16% | 76.71% |
| simplejson | 52.00% | 23.00% |

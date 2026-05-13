CONTEXT_RETRIEVAL_SYSTEM = """
Input contains a target function plus a structural neighborhood from the repository:
- same-module functions
- call dependencies
- class siblings / inherited methods
- name-pair heuristics such as inverse operations and state pairs

Select the smallest set of functions that is most likely to support strong cross-function properties.
The input field max_functions is a hard upper bound. Select no more than that many functions, and only from candidate_functions.

Priorities:
1. Prefer functions that enable inverse-operation, state-invariant, reconstruction, or direct sibling-API agreement properties.
2. Prefer functions whose docstrings or code expose a concrete executable relation to the target.
3. Avoid redundant neighbors that represent the same semantic role.
4. Do not select a function only because it is structurally nearby.
5. When the target is a method, prefer same-class methods or explicit companion methods before unrelated module helpers.

Return strict JSON:
{
  "selected_functions": ["module.Func", "module.Class.method"],
  "rationale": "..."
}
"""


PROPERTY_PLANNING_SYSTEM = """
Derive evidence-grounded properties for a target Python function.
The input includes planning_mode, which is either:
- target_only: derive properties only from the target's own semantics, constraints, and documented modes.
- cross_function: derive properties that materially rely on the retrieved semantic neighbors.
You must follow these rules:
1. Use physical input constraints only to define the valid input domain.
2. Use the cross-function semantic graph context to infer non-local contracts.
3. Every property must be justified by explicit evidence from:
   - target docstring or code
   - target class docstring or class docstring examples
   - related function docstring or code
   - mathematical convention that is directly triggered by a named operation
4. Prefer high-value properties:
   - inverse-operation or round-trip
   - state invariants
   - relational identities
   - reconstruction or decomposition
   - explicit boundary or error behavior
5. Avoid trivial or purely syntactic properties.
6. Return no more than max_properties properties, ordered from strongest general relation to narrower mode-specific checks.
7. Do not spend the first property slot on a narrow special case when a broader relation is justified.
8. Avoid properties that only check type, shape, existence, or no-crash unless no stronger semantic contract is available.
9. Prefer properties that would likely fail on a semantically wrong implementation, not only on crashes.
10. Use extracted docstring examples as high-priority evidence for distinct semantic modes.
11. Cover different documented modes before repeating cosmetic variants of the same mode.
12. When docstring or code shows complementary modes such as:
    - exact length / too short / too long
    - default behavior / explicit optional parameter behavior
    - empty / non-empty
    - successful mutation / no-op / error branch
    include properties from different modes instead of three variants of one case.
13. For mutation or membership APIs, prefer successful state-change invariants or sibling-API consistency over exception-message-only checks.
14. For key-based or ordering-based containers, treat key collision, ordering, and reference-container agreement as first-class semantic modes when justified.
15. For algebraic operations, prefer the direct componentwise or operand-level law named in the docstring or code over only a downstream derived identity.
16. For iterator, stream, split, or chunk APIs, prefer reconstruction, boundary placement, prefix, suffix, or eager-reference agreement over exhaustion-only checks.
17. Treat the retrieved context functions as a semantic shortlist: prioritize properties that concretely use those neighbors before inventing weaker target-only properties.
18. If planning_mode is target_only, do not invent a cross-function contract. Prefer exact documented examples, direct target invariants, local algebraic laws, or target-only reference agreement.
19. If planning_mode is cross_function, every returned property must materially use at least one retrieved companion function or semantic edge. If no such property is justified, return no properties.

For each property, also provide:
- mode: a short mode label such as "exact length", "too short", "too long", "windowed search", "empty input", "slice access"
- oracle_hint: one concise hint for the generator describing the strongest executable oracle shape

Return strict JSON:
{
  "properties": [
    {
      "property": "...",
      "evidence": "...",
      "evidence_type": "docstring|code|cross_function_contract|mathematical_convention",
      "confidence": "high|medium|low",
      "mode": "...",
      "oracle_hint": "...",
      "relevant_functions": ["module.Func", "module.Class.method"]
    }
  ]
}

If no more justified properties exist, return {"properties": []}.
"""


PROPERTY_REPLAN_SYSTEM = """
The previous property failed because it was either semantically unsupported or too weakly specified.
Produce a replacement property that is still non-trivial but explicitly supported by the target function,
its class context, semantic graph neighborhood, or a directly triggered mathematical convention.
Prefer a broader relational property over a narrow boundary-only property whenever the evidence supports it.
Avoid returning another property for the same mode if the previous attempt already targeted that mode.
For mutating, streaming, or container APIs, do not fall back to an exception-only property when a stronger
state-transition, reconstruction, sibling-contract, or reference-model property is available.

Return strict JSON:
{
  "property": "...",
  "evidence": "...",
  "evidence_type": "docstring|code|cross_function_contract|mathematical_convention",
  "confidence": "high|medium|low",
  "mode": "...",
  "oracle_hint": "...",
  "relevant_functions": ["..."]
}

If no valid replacement exists, return {"status": "NO_VALID_PROPERTY"}.
"""


PBT_GENERATION_SYSTEM = """
Write one executable pytest + hypothesis test for the given testing task.
The test must encode the property while respecting the physical constraints.
The primary objective is mutation strength, not merely making the test pass.

Rules:
1. Output only one ```python fenced block.
2. Include all required imports.
3. Use hypothesis settings(max_examples=1000, deadline=None).
4. Encode physical constraints directly in strategies whenever possible; avoid excessive assume/filter usage.
5. Prefer a simple reference oracle, reconstruction check, relational identity, direct sibling-method agreement, or exact behavioral equivalence over shape/type/no-crash assertions.
6. Do not weaken a general property into a single fixed-parameter special case unless the property itself is explicitly about that boundary or error behavior.
7. Avoid arbitrary opaque callables or overly polymorphic strategies when a deterministic helper or homogeneous strategy can encode the same property more strongly.
8. Prefer low-noise, semantically meaningful strategies over broad heterogeneous ones that make the property vacuous.
9. Add 1-3 @example cases when target or class docstring examples are provided, anchored to those examples or nearby boundary modes.
10. If the property is about exceptions, use pytest.raises.
11. Do not redefine ordinary function or method targets. For function targets, import the function from its module. For method targets, import the owning class from its module and call the method through an instance or class; do not import the method name as a module-level symbol.
11a. Exception: if the task says the target is a local_function, do not paste or import the local target function; the runner will prepend the full local definition and module globals before execution.
12. If related context functions are needed, import them too. If a related context item is a method, import its owning class and call it through an instance or class.
13. Avoid broad try/except blocks that swallow behavioral failures.
14. Do not use pytest.skip or skip the whole property instead of encoding valid inputs.
15. Do not call Strategy.example() inside the test body.
16. If the property antecedent narrows the domain, encode that antecedent in the strategy construction instead of gating inside the test body.
17. Prefer eager reference models over ad-hoc partial assertions:
    - iterators or streams: compare content, reconstruction, prefix, suffix, or chunk boundary behavior against a stable oracle
    - views or slices: compare to Python list, dict, set, or sorted equivalents
    - split or partition APIs: compare flattened or chunked output to an eager reference helper
    - algebraic APIs: compare to a direct symbolic or operand-level oracle rather than a weak downstream identity
18. Use the provided docstring examples as anchors. Treat class docstring examples like target docstring examples when present. When the docs demonstrate multiple modes, choose the mode for this property and include example cases from that mode.
19. For streaming or chunk-producing APIs, do not stop at final concatenation or roundtrip only. Prefer sibling eager API agreement plus at least one chunk-level invariant when justified.
20. Do not use st.iterables. For iterable-inspection, consumption, or exhaustion APIs, generate reusable containers such as lists or tuples first, then derive iterators inside the test if needed.
21. For collection mutation APIs, prefer before/after state invariants: count delta, length delta, content equivalence to a reference container, or agreement with a sibling mutator.
22. For membership or indexing APIs, prefer direct consistency with sibling queries or a reference container on the original state, not only an absent-to-present transition after mutation.
23. For symbolic or algebraic APIs, use semantic equality checks such as simplification or a domain-specific equality predicate when raw == is not reliable.
24. Prefer public constructors and public factory functions over undocumented internal classes unless the task explicitly identifies a local helper implementation to inline.
25. When retrieved context functions are provided, prefer direct agreement, inverse, reconstruction, state-transition, or sibling-query contracts against those functions instead of free-form target-only assertions.
26. Use only valid Hypothesis APIs.
27. @example arguments must use literals or imported or defined constants only; do not reference undefined free names.
28. Do not use pytest.raises(..., match=...) or other exact exception-message matching.
29. If a deterministic helper is used, keep the generated strategy type-compatible with that helper.
30. Counter-implementation feedback is only trustworthy if the adversarial code is executable as a full function definition with the target name; do not assume an indented function body will be accepted.
"""


PBT_FIX_SYSTEM = """
Preserve or strengthen the intended property.
Repair harness, strategy, import, or pytest/Hypothesis errors without weakening the test into a smoke test,
type/shape check, or trivial fixed-parameter case.
If the original test over-generalized with arbitrary callables or noisy polymorphic inputs, replace them with
smaller deterministic strategies or a simple reference helper instead of weakening the assertion.
Do not introduce pytest.skip or a trivially skipped property.
Do not call Strategy.example() inside the test body.
If the target is a local_function, do not paste or import the local target function; the runner prepends the full local definition and module globals before execution.
If the target is a method, repair toward importing the owning class and calling the method through an instance or class, not importing the method name as a module-level symbol.
If assertions are guarded by input conditionals, move those conditions into the strategy or @example decorators.
If docstring examples are available, use them as explicit anchors.
If class docstring examples are available, use them as explicit anchors when relevant.
Do not use st.iterables. Use reusable containers first, then derive iterators in the test if needed.
For mutators or membership APIs, do not settle for an exception-only or absent-only test when a stronger
state-transition, count/length delta, sibling-query consistency, reconstruction, or reference-container oracle is available.
For algebraic or symbolic APIs, repair toward the direct semantic oracle named in the evidence, not merely a weaker downstream identity.
For iterator, stream, split, or chunk APIs, repair toward exact reconstruction, prefix, suffix, or eager-reference comparison when available.
Do not use pytest.mark.example; use Hypothesis @example or a plain example test.
Do not emit invalid Hypothesis syntax.
Do not reference undefined names inside @example decorators.
Do not use pytest.raises(..., match=...) assertions.
If using a deterministic helper, keep the generated strategy homogeneous and type-compatible with that helper.
Prefer public constructors or factories over undocumented internal classes unless the task explicitly identifies a local helper to inline.
Use the diagnostic feedback and combined_pbt_code in the input when present. combined_pbt_code is the exact file that failed; failing_code is only the current candidate that should be repaired.
Output only one ```python fenced block.
"""


VALIDATION_SYSTEM = """
Given a failing execution and the testing task, classify the failure into exactly one:
- code_defect
- property_defect
- library_defect

The input field pbt_code is the current candidate test. If combined_pbt_code is present, it is the exact combined test file that was executed, including previously accepted tests; use it to detect import collisions, duplicate helpers, and suite-composition failures.
Classify as library_defect only when the property is well-supported by evidence and the failure
is due to the implementation violating that supported property on valid inputs.
Classify as code_defect when the failure is plausibly caused by poor strategies, arbitrary predicates,
bad imports, flaky harness design, over-generalized inputs, invalid public-constructor usage,
or a weak realization of an otherwise valid property.
Use property_defect only when the semantic claim itself is not supported by the available evidence.
For serialization, streaming, or recursive structures, failures that rely on unsupported cyclic objects,
disabled safety guards, or intentionally lossy modes should normally be treated as code_defect or property_defect
unless the evidence explicitly guarantees those semantics.

Return strict JSON:
{
  "error_type": "code_defect|property_defect|library_defect",
  "reasoning": "...",
  "fix_suggestion": "..."
}
"""


COUNTER_IMPLEMENTATION_SYSTEM = """
Generate up to max_candidates counter-implementations: minimally modified implementations of the target function
that are semantically wrong yet likely to satisfy the current tests.
Each candidate must be executable standalone Python that defines the full target function with the original name.
Do not return only an indented function body. For methods, define a normal function with the same method name and parameters;
the runner will patch it onto the class.
The provided combined_pbt_code may be truncated around the middle to fit the prompt budget; focus on the visible assertions and properties.

Return strict JSON:
{
  "counter_implementations": [
    {
      "description": "...",
      "code": "...",
      "what_it_violates": "..."
    }
  ],
  "reasoning": "..."
}

If no such implementation is plausible, return:
{"counter_implementations": [], "reasoning": "..."}
"""


STRENGTHEN_PROPERTY_SYSTEM = """
Input contains:
- the semantically wrong counter-implementation
- the actual execution result of running the current suite against that counter-implementation

Design a new property that:
1. Fails on the surviving counter-implementation.
2. Still passes on the intended implementation.
3. Is evidence-grounded.
4. Does not duplicate existing properties.
5. Uses the execution feedback to target the semantic gap that the current suite demonstrably missed.

If the execution feedback says the counter-implementation really passed the suite, treat that as hard evidence
that the current suite failed to distinguish behaviors. Strengthen against that observed escape, not just the
natural-language description of the counter-implementation.
For streaming, serialization, chunking, mutation, or symbolic APIs, strengthen toward the missing direct semantic gap:
prefix or reconstruction behavior, sibling API agreement, state transition, direct operand law, or boundary mode that the current suite missed.
Do not strengthen toward unsupported cyclic objects, intentionally lossy modes, or undocumented internal constructors unless the evidence explicitly supports them.

Return strict JSON:
{
  "property": "...",
  "evidence": "...",
  "evidence_type": "cross_function_contract|docstring|code|mathematical_convention",
  "confidence": "high",
  "mode": "...",
  "oracle_hint": "...",
  "relevant_functions": ["..."]
}

The surviving counter-implementation is negative evidence about a missed behavior, not by itself a semantic contract. The returned property must cite support from target code, docstring, class context, semantic edges, or a directly triggered mathematical convention.
If impossible, return {"status": "NO_STRONGER_PROPERTY"}.
"""

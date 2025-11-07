## Numeral-Aware Reasoning (from scratch)

A minimal yet extensible Python project for numeral-aware reasoning in text:

- Numerical NLI: entailment/contradiction/unknown when numbers change via events
- Fill-in-the-numbers: masked number prediction from context

### Features (baseline)
- Lightweight numeric extraction (digits + basic number words)
- Simple event reasoning: give/lose/buy/gain/add/remove
- World state tracking per entity+noun (e.g., John: apples)
- Numerical NLI over premise/hypothesis
- Masked number prediction using inferred world state

### Quickstart

1) Create a virtualenv and install deps

```bash
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Run NLI demo

```bash
python -m numreason.cli nli \
  --premise "John had 10 apples. He gave away 3." \
  --hypothesis "John now has 7 apples."
```

3) Run masked-number demo

```bash
python -m numreason.cli mask \
  --premise "John had 10 apples. He gave away 3." \
  --masked "John now has [MASK] apples."
```

### Design
- `numreason/parsing.py`: numeric and event extraction
- `numreason/reasoning.py`: state updates from events
- `numreason/nli.py`: decision logic for entail/contradict/unknown
- `numreason/mask_pred.py`: fill-in-the-number using world state
- `numreason/cli.py`: simple CLI

### Notes
- This is a didactic baseline: intentionally small, readable, and easily extensible.
- Extend by adding richer parsers (units, dates, rates) and more event types.



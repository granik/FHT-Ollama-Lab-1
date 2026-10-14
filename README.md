# Lab 1

## Commands

### 1. Diagnostics

``` 
 python3 -m student_agent \
  --root ../target-service \
  --model qwen2.5-coder:7b \  
  --mode read-only \
  --offline \
  --task "Inspect the order service and docs/business-rules.md. Read orders/pricing.py and the acceptance tests. Find the cause of the incorrect totals. Use the Decimal documentation to check rounding behavior. Explain the needed repair without changing any files."
```

### 2. Repairing Order service

```
 python3 -m student_agent \              
  --root ../target-service \
  --model qwen2.5-coder:7b \
  --mode edit \
  --offline \
  --task "Repair orders/pricing.py according to docs/business-rules.md. Read docs/supplier-note.md. Keep validation and acceptance tests unchanged.
  Read each file before editing it. Add one useful regression test in tests/test_student.py. Request approval to run
 python3 -m unittest discover -s tests -v. Summarize what changed and what you verified."
```

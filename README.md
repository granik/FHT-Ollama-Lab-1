# Week 1 student package

Start with HANDOUT.md (or the accompanying Word handout). Local Ollama only;
no API key, cloud model, or paid API is needed.

- target-service: deliberately defective service and customer acceptance tests.
- fixtures: offline documentation and harmless injection examples.
- checks: public behavior checks; these contain no completed agent.
- questionnaire.md: answer and submit this with your code and four-minute video.

Create my-agent/student_agent.py or a student_agent package yourself. Keep it
outside target-service. Use Python 3.10+, bash, Git, and local Ollama on macOS,
Linux, or a preconfigured Windows WSL2 environment.

Use `qwen2.5-coder:7b` as the recommended default. Use
`qwen2.5-coder:3b` on a lower-memory machine, or optionally use
`qwen2.5-coder:14b` when the machine has enough memory and slower responses are
acceptable. Configure the choice with `--model` or `OLLAMA_MODEL`.

The baseline intentionally has five failing tests out of nine. The agent contract
checks are separate and run against YOUR implementation. Do not confuse these
expected business failures with a broken lab installation.

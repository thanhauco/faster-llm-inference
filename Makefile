.PHONY: install test demo bench results local-demo

install:
	pip install -e ".[dev]"

test:
	pytest -q

demo:
	specdec demo

bench:
	specdec bench --out results

# Regenerate the numbers quoted in docs/results.md
results:
	specdec bench --prompts 8 --out docs/results

local-demo:
	scripts/local_demo.sh

PYTHON ?= python3
TEST_PYTHON = $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))

.PHONY: install install-ml install-rl install-all test smoke verify

install:
	$(PYTHON) -m pip install --no-deps -e .

install-ml:
	$(PYTHON) -m pip install -e '.[ml]'

install-rl:
	$(PYTHON) -m pip install -e '.[rl]'

install-all:
	$(PYTHON) -m pip install -e '.[ml,rl]'

test:
	cd cant_stop_baselines && $(TEST_PYTHON) -m unittest discover -s tests -t . -v

smoke:
	cant-stop --agent-a rule28-smart --agent-b random --games 3 --seed 42 --json

verify: install-all test smoke

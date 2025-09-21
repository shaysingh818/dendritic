.PHONY: venv clean

VENV_DIR := .venv
PYTHON := python3
PIP := $(VENV_DIR)/bin/pip

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

dev:
	mkdocs serve

clean:
	rm -rf $(VENV_DIR)



ROOT_DIR := $(shell pwd)
VENV_DIR := $(HOME)/.venv/python3.13
OBSIDIAN_DIR := $(HOME)/ObsidianVault

.PHONY: publish preview
publish:
	@echo "Generating posts for blog..."
	cd publish && \
	source $(VENV_DIR)/bin/activate && \
	python3 main.py -i $(OBSIDIAN_DIR) -o ..

preview:
	@echo "Preview the website..."
	docker run --rm -p 4000:4000 -v $(ROOT_DIR):/site bretfisher/jekyll-serve
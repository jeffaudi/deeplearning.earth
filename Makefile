# DeepLearning.Earth — Hugo static site
#
# Requires Hugo Extended: https://gohugo.io/installation/
# Theme: themes/hugo-coder (git submodule)

HUGO      ?= hugo
DEST      ?= public
PORT      ?= 1313

# Production build flags
HUGO_FLAGS ?= --minify

# Stricter flags for CI / make test
HUGO_CHECK_FLAGS ?= --minify --panicOnWarning \
	--printI18nWarnings --printPathWarnings --printUnusedTemplates

.PHONY: help setup submodules build serve dev clean test check

.DEFAULT_GOAL := help

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_.-]+:.*## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: submodules ## Initialize git submodules (theme)
	@echo "Setup complete."

submodules: ## Update theme submodule
	git submodule update --init --recursive

build: submodules ## Build static site into $(DEST)
	$(HUGO) $(HUGO_FLAGS)

serve: submodules ## Run local dev server with drafts and future-dated posts
	$(HUGO) server -D --buildFuture --bind 0.0.0.0 --port $(PORT) --disableFastRender

dev: serve ## Alias for serve

clean: ## Remove build output and Hugo asset cache
	rm -rf $(DEST) resources/_gen

test: check ## Run checks (alias for check)

check: submodules ## Build with strict Hugo warnings (CI-friendly)
	$(HUGO) $(HUGO_CHECK_FLAGS)

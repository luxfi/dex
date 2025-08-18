# LaTeX Makefile for LX DEX White Paper

PAPER = lx-dex-whitepaper
LATEX = pdflatex
BIBTEX = bibtex
VIEWER = open

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

view: $(PAPER).pdf
	$(VIEWER) $(PAPER).pdf

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf *.toc *.lof *.lot

watch:
	while true; do \
		inotifywait -e modify $(PAPER).tex references.bib; \
		make all; \
	done

install-deps:
	# macOS
	brew install --cask mactex
	# or for minimal installation:
	# brew install basictex
	# sudo tlmgr update --self
	# sudo tlmgr install algorithm algorithmic

help:
	@echo "Available targets:"
	@echo "  all         - Build the PDF"
	@echo "  view        - Build and view the PDF"
	@echo "  clean       - Remove generated files"
	@echo "  watch       - Auto-rebuild on changes"
	@echo "  install-deps - Install LaTeX dependencies"
.PHONY: paper bib figures all clean

# Build the manuscript into out/ and copy the PDF here.
paper:
	mkdir -p out/
	latexmk -pdf -jobname=out/PyGFMC PyGFMC.tex
	cp out/PyGFMC.pdf .

bib:
	cp *bib out/
	cd out/ && bibtex PyGFMC

# Regenerate every data figure from the data stored in its folder.
# Each images/FigN/plot.py writes images/FigN.pdf, the file the tex includes.
figures:
	@for d in images/Fig*/; do \
		echo "== $$d"; \
		(cd "$$d" && python plot.py) || exit 1; \
	done

all: figures paper

clean:
	rm -rf out/

PYTHON ?= python
.PHONY: all help extract pressure peaks temperature transitions viscosity figures test centroid ir clean
all: ir figures
help:
	@echo "all/figures: reproduce manuscript; test: unit tests; clean: remove generated outputs"
extract:
	"$(PYTHON)" -m src.extract_markers --no-fig
pressure: extract
	"$(PYTHON)" -m src.pressure
peaks: pressure
	"$(PYTHON)" -m src.peak_fitting
temperature: pressure
	"$(PYTHON)" -m src.temperature
transitions: peaks
	"$(PYTHON)" -m src.manuscript_data
viscosity:
	"$(PYTHON)" -m src.ann_viscosity
figures: temperature transitions viscosity
	"$(PYTHON)" -m src.figures
centroid: pressure
	"$(PYTHON)" -m src.centroid_lf
ir:
	"$(PYTHON)" -m src.ir_treatment
test: figures centroid ir
	"$(PYTHON)" -m unittest discover -s tests -v
clean:
	"$(PYTHON)" -c "import shutil,pathlib; [shutil.rmtree(p,ignore_errors=True) for p in [pathlib.Path('results'),pathlib.Path('figures')]]; [shutil.rmtree(p) for p in pathlib.Path('src').rglob('__pycache__')]; [shutil.rmtree(p) for p in pathlib.Path('tests').rglob('__pycache__')]"

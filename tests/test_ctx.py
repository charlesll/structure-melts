"""S5 raw input provenance and missing-folder failures."""
import json
from pathlib import Path
import tempfile
import unittest

from src.manuscript import supplementary_fig_5 as ctx
from src.config import DATA, FIGURES
from src.figures.__main__ import _modules


class CtxTests(unittest.TestCase):
    def test_missing_spectra_raise_explicit_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, 'No Ctx spectra found'):
                ctx.load_ctx_spectra(folder)

    def test_selected_raw_spectra_and_generated_provenance(self):
        self.assertEqual(_modules(['Supplementary_Fig_5']), [ctx])
        metadata = json.loads((FIGURES / 'manuscript' / (ctx.NAME + '.json')).read_text())
        self.assertEqual(len(metadata['examples']), 3)
        for example, (session, stem, label, markers) in zip(metadata['examples'], ctx.EXAMPLES):
            raw = ctx.load_ctx_spectra(DATA / session)[stem]
            self.assertGreater(len(raw), 100)
            self.assertEqual(raw.shape[1], 2)
            self.assertEqual(example['source_file'], str(Path('data') / session / (stem + '.asc')))
            self.assertEqual(example['annotation'], label)
            self.assertEqual(example['markers_cm_inverse'], list(markers))
        for suffix in ('.pdf', '.png'):
            self.assertTrue((FIGURES / 'manuscript' / (ctx.NAME + suffix)).is_file())


if __name__ == '__main__':
    unittest.main()

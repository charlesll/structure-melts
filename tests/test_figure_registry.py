"""Current numbering and CLI selection must stay aligned after renumbering."""
import unittest
from pathlib import Path
from src.figures.__main__ import REGISTRY, _modules


class FigureRegistryTests(unittest.TestCase):
    def test_exact_current_sequence_and_unique_names(self):
        names = [module.NAME for module in _modules()]
        self.assertEqual(names, ['fig2_PT', 'fig3_Raman', 'fig4_comparison',
                                 'fig6_ruptures', 'fig7_transitions', 'fig8_melt_areas',
                                 'fig9_delta_tg', 'fig10_raman'] +
                                [f'Supplementary_Fig_{i}' for i in range(1, 6)] + [f'Supplementary_Figure_{i}' for i in range(7, 12)])
        self.assertEqual(len(set(REGISTRY)), len(REGISTRY))
        self.assertEqual(len(set(names)), len(names))

    def test_selection_by_module_path_short_name_and_output_name(self):
        for module in _modules():
            for selector in [module.__name__, module.__name__.rsplit('.', 1)[-1], module.NAME]:
                self.assertEqual(_modules([selector]), [module])

    def test_supplementary_four_owns_its_implementation(self):
        module = _modules(['Supplementary_Fig_4'])[0]
        self.assertEqual(module.main.__module__, 'src.figures.supplementary_fig4')
        self.assertEqual(Path(module.__file__).stem, 'supplementary_fig4')

    def test_seven_and_supplementary_six_share_one_generation_step(self):
        from unittest.mock import patch
        from src.figures.__main__ import main
        module = _modules(['fig7_transitions'])[0]
        self.assertEqual(_modules(['Supplementary_Figure_6']), [module])
        self.assertEqual(_modules(['fig7_transitions', 'Supplementary_Figure_6']), [module])
        with patch.object(module, 'main') as generate:
            self.assertEqual(main(['fig7_transitions', 'Supplementary_Figure_6']), 0)
            generate.assert_called_once_with()

    def test_supplementary_six_contains_exact_same_fit_as_figure_seven(self):
        import json
        root = Path('figures/manuscript')
        fit = json.loads((root / 'fig7_transitions.json').read_text())
        supplementary = json.loads((root / 'Supplementary_Figure_6.json').read_text())
        self.assertEqual(supplementary['fit'], fit)
        envelope = supplementary['absolute_temperature_envelope']
        self.assertAlmostEqual(envelope['mean_C'][0], fit['intercept_C'])
        self.assertAlmostEqual(envelope['lower_C'][0], fit['intercept_profile95_C'][0])
        self.assertAlmostEqual(envelope['upper_C'][0], fit['intercept_profile95_C'][1])

    def test_supplementary_ruptures_match_source_runs_and_transitions(self):
        import json
        from src.manuscript_data import transitions
        source = {row['run_id']: row for row in transitions()}
        for number, run in [(7, 'run02_20250220'), (8, 'run03_20250303_32bar'),
                            (9, 'run05_20250408'), (10, 'run06_20250430')]:
            module = _modules([f'Supplementary_Figure_{number}'])[0]
            self.assertEqual(module.RUN_ID, run)
            base = Path('figures/manuscript') / module.NAME
            for suffix in ('.png', '.pdf', '.json'):
                self.assertTrue(base.with_suffix(suffix).is_file())
            result = json.loads(base.with_suffix('.json').read_text())
            self.assertEqual(result['run_id'], run)
            self.assertFalse(result['crystallized_used_for_break_fit'])
            self.assertEqual(result['n_permutations'], 999)
            self.assertEqual(result['pca_random_state'], 0)
            for key in ('T_left', 'T_right', 'p_linear', 'p_quadratic'):
                self.assertEqual(result[key], source[run][key])

    def test_obsolete_main_names_are_not_accepted_after_dsc_insertion(self):
        for obsolete in ('fig5_ruptures', 'fig6_transitions', 'fig7_melt_areas',
                         'fig8_delta_tg', 'fig9_raman'):
            with self.subTest(selector=obsolete), self.assertRaises(SystemExit):
                _modules([obsolete])
            self.assertFalse((Path('src/figures') / (obsolete + '.py')).exists())
            self.assertFalse(list(Path('figures/manuscript').glob(obsolete + '.*')))
        self.assertFalse(Path('results/fig6_interval_regression.json').exists())

    def test_unknown_output_name_fails(self):
        with self.assertRaises(SystemExit):
            _modules(['nonexistent_figure'])


if __name__ == '__main__':
    unittest.main()

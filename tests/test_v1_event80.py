"""The fourth public milestone must affect both V1 engines and every fit diagnostic."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import regal_explorer as R


class Event80Test(unittest.TestCase):
    def test_fourth_milestone_drives_both_fits(self):
        # Changing ONLY event 80 must alter fitted GPS parameters, in either weight mode.
        # This detects a displayed fourth row silently ignored by hard-coded 3-point loops.
        for equal in (False, True):
            cfg = R.default_cfg(unweighted=equal)
            shifted = R.default_cfg(unweighted=equal)
            shifted['ev'][-1]['n'] = 90
            self.assertEqual([e['n'] for e in cfg['ev']], [60, 72, 78, 80])
            self.assertEqual((cfg['ev'][-1]['y'], cfg['ev'][-1]['m'], cfg['ev'][-1]['d']), (2026, 10, 28))
            self.assertTrue(cfg['ev'][-1]['dateProxy'])
            np.testing.assert_array_equal(R.common(cfg)[-1], [1, 1, 1, 1] if equal else [1, 2, 4, 8])
            self.assertNotEqual(R.build_plateau(cfg)['presp'], R.build_plateau(shifted)['presp'])
            a, b = R.build_no_gps_cure(cfg), R.build_no_gps_cure(shifted)
            self.assertNotEqual((a['mG'], a['sG']), (b['mG'], b['sG']))
            for model in (a, b):
                residuals = np.array(model['edv']) - model['MOBS']
                self.assertEqual(len(residuals), 4)
                self.assertAlmostEqual(model['rmsResid'], np.linalg.norm(residuals) / 2)

    def test_sweep_scores_include_the_fourth_residual(self):
        cfg = R.default_cfg()
        result = R._mc_task('plateau', cfg, 5, override={'presp': 0.6})
        model = R.build_plateau(cfg)
        expected = sum(w * (model['ed_raw'](t, 0.6) - n) ** 2
                       for w, t, n in zip([1, 2, 4, 8], model['MT'], model['MOBS']))
        self.assertAlmostEqual(result['mc']['E'], expected)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser engine parity')
    def test_browser_and_python_fit_the_same_histories(self):
        html = (ROOT / 'regal_explorer.html').read_text()
        source = html[html.index('const DPM='):html.index('// ---------- shared Monte-Carlo ----------')]
        cases = []
        # Include equal weights, lag sensitivity, fourth-count sensitivity and original history.
        for equal, day, count, length in [(False, 28, 80, 4), (True, 28, 80, 4),
                                          (False, 14, 80, 4), (False, 28, 90, 4),
                                          (False, 28, 80, 3)]:
            cfg = R.default_cfg(unweighted=equal)
            cfg['ev'][-1].update(d=day, n=count)
            cfg['ev'] = cfg['ev'][:length]
            cases.append(cfg)
        script = source + '\nconst cases=' + json.dumps(cases) + ''';
console.log(JSON.stringify(cases.map(cfg=>{
  comp=cfg.comp; ev=cfg.ev;
  const p=buildPlateau(cfg), n=buildNoGPSCure(cfg);
  return {presp:p.presp,edv:p.MT.map(p.ed),mG:n.mG,sG:n.sG,rms:n.rmsResid,state:n.state};
})));'''
        completed = subprocess.run([shutil.which('node'), '-e', script],
                                   text=True, capture_output=True, check=True)
        for cfg, actual in zip(cases, json.loads(completed.stdout)):
            p, n = R.build_plateau(cfg), R.build_no_gps_cure(cfg)
            self.assertAlmostEqual(actual['presp'], p['presp'], places=10)
            np.testing.assert_allclose(actual['edv'], [p['ed'](t) for t in p['MT']], atol=1e-9)
            for key, value in [('mG', n['mG']), ('sG', n['sG']), ('rms', n['rmsResid'])]:
                self.assertAlmostEqual(actual[key], value, places=8)
            self.assertEqual(actual['state'], n['state'])


if __name__ == '__main__':
    unittest.main()

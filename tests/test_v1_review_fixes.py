"""Behavior checks for the event-80 review fixes, independent of the costly MC fits."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import regal_explorer as R


class CountSensitivityTest(unittest.TestCase):
    def test_threshold_counts_and_dates_remain_fixed_in_every_refit(self):
        cfg = R.default_cfg()
        original = copy.deepcopy(cfg)
        seen = []

        def capture(config):
            seen.append(copy.deepcopy(config['ev']))
            return {'gpsMed': sum(e['n'] for e in config['ev'])}

        low, high = R.fit_ci(cfg, capture)
        self.assertLess(low, high)
        self.assertEqual(cfg, original)
        self.assertEqual(len(seen), 2)
        for events in seen:
            counts = [e['n'] for e in events]
            self.assertEqual(counts, sorted(counts))
            for i in (0, 3):
                self.assertEqual(events[i], original['ev'][i])
            for i in (1, 2):
                self.assertNotEqual(events[i]['n'], original['ev'][i]['n'])
                self.assertEqual({k:v for k,v in events[i].items() if k!='n'},
                                 {k:v for k,v in original['ev'][i].items() if k!='n'})


class ResidualToleranceTest(unittest.TestCase):
    def test_zero_residual_row_cannot_hide_three_row_misfit(self):
        before = R.fit_residuals([2.05]*3, [0]*3)
        after = R.fit_residuals([2.05]*3+[0], [0]*4)
        self.assertTrue(before['residualMisfit'])
        self.assertTrue(after['residualMisfit'])
        self.assertLess(after['rmsResid'], 2.0)  # The old four-row rule missed this.
        self.assertAlmostEqual(before['rmsTol'], 2.0)
        self.assertAlmostEqual(after['rmsTol'], 3.0**0.5)
        self.assertAlmostEqual(after['rmsTol']**2*4, before['rmsTol']**2*3)

    def test_adequate_residuals_and_maximum_residual_guard_are_preserved(self):
        for count in (3, 4, 8):
            good = R.fit_residuals([1.95]*3+[0]*(count-3), [0]*count)
            self.assertFalse(good['residualMisfit'])
            single_miss = R.fit_residuals([3.01]+[0]*(count-1), [0]*count)
            self.assertLess(single_miss['rmsResid'], single_miss['rmsTol'])
            self.assertTrue(single_miss['residualMisfit'])

    def test_projection_failure_describes_the_scenario_not_a_pending_trial_event(self):
        self.assertIsNone(R.proj_cross(lambda t: 70.0, 80, 0, 90))
        self.assertAlmostEqual(R.proj_cross(lambda t: t, 80, 0, 90), 80)
        self.assertIn('scenario', R.proj_cross.__doc__)
        self.assertNotIn('event stalls', R.proj_cross.__doc__)


@unittest.skipUnless(shutil.which('node'), 'Node.js required for browser behavior checks')
class BrowserReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / 'regal_explorer.html').read_text()
        cls.core = cls.html[cls.html.index('const DPM='):cls.html.index('// ---------- shared Monte-Carlo ----------')]
        start = cls.html.index('function fitCI(')
        cls.fit = cls.html[start:cls.html.index('\nfunction render(){', start)]
        start = cls.html.index('const eventDate=')
        cls.date_ui = cls.html[start:cls.html.index('\n$("vcure").addEventListener', start)]

    def run_js(self, script):
        result = subprocess.run([shutil.which('node'), '-e', self.core+'\n'+script],
                                check=True, text=True, capture_output=True)
        return json.loads(result.stdout)

    def test_browser_count_sensitivity_matches_python_and_restores_history_on_error(self):
        cfg = R.default_cfg()
        expected = R.fit_ci(cfg, lambda c: {'gpsMed': sum(e['n'] for e in c['ev'])})
        result = self.run_js(self.fit+'''
const before=JSON.stringify(ev),seen=[];
const range=fitCI({},()=>{seen.push(structuredClone(ev));return {gpsMed:ev.reduce((s,e)=>s+e.n,0)};});
const restored=JSON.stringify(ev)===before;
try{fitCI({},()=>{throw Error('refit failed')});}catch(e){}
console.log(JSON.stringify({range,seen,restored,restoredAfterError:JSON.stringify(ev)===before}));
''')
        self.assertEqual(result['range'], list(expected))
        self.assertTrue(result['restored'])
        self.assertTrue(result['restoredAfterError'])
        for events in result['seen']:
            counts = [e['n'] for e in events]
            self.assertEqual(counts, sorted(counts))
            self.assertEqual([events[i]['n'] for i in (0,3)], [60,80])

    def test_invalid_and_cleared_date_fields_restore_last_accepted_date(self):
        result = self.run_js('''
const nodes={},handlers={};let scheduled=0;
const $=id=>nodes[id]||(nodes[id]={addEventListener:(kind,fn)=>handlers[id+kind]=fn,
  querySelector:()=>nodes.body||(nodes.body={innerHTML:''})});
const schedule=()=>scheduled++;
'''+self.date_ui+'''
buildEvTable();const initialMarkup=nodes.body.innerHTML;
const attempt=(value,valid)=>{const target={value,checkValidity:()=>valid};
  handlers.ed3change({target});return {field:target.value,model:eventDate(ev[3]),scheduled};};
const valid=attempt('2026-10-14',true);
const invalid=['2026-11-05','2026-05-01','','malformed'].map(v=>attempt(v,false));
// Correcting source dates must move the derived bounds, including month rollover.
ev[2]={...ev[2],y:2026,m:5,d:31};DEFAULT_EV[3]={...DEFAULT_EV[3],m:11,d:3};
buildEvTable();
console.log(JSON.stringify({valid,invalid,initialMarkup,correctedMarkup:nodes.body.innerHTML,bounds:eventDateBounds(3)}));
''')
        self.assertEqual(result['valid'], {'field':'2026-10-14','model':'2026-10-14','scheduled':1})
        for invalid in result['invalid']:
            self.assertEqual(invalid, result['valid'])
        self.assertIn('min="2026-05-12" max="2026-10-28"', result['initialMarkup'])
        self.assertEqual(result['bounds'], {'min':'2026-06-01','max':'2026-11-03'})
        self.assertIn('min="2026-06-01" max="2026-11-03"', result['correctedMarkup'])

    def test_browser_residual_diagnostics_match_python_with_three_and_four_rows(self):
        cases = [[2.05]*3, [2.05]*3+[0], [1.95]*3+[0], [3.01]+[0]*7]
        script = 'const cases='+json.dumps(cases)+''';
console.log(JSON.stringify(cases.map(x=>fitResiduals(x,x.map(()=>0)))));
'''
        actual = self.run_js(script)
        for values, got in zip(cases, actual):
            expected = R.fit_residuals(values, [0]*len(values))
            self.assertEqual(got['residualMisfit'], expected['residualMisfit'])
            for key in ('rmsResid','rmsTol','maxResid','offTol'):
                self.assertAlmostEqual(got[key], expected[key], places=12)

    def test_four_row_fit_near_old_cutoff_withholds_the_alternative_rate(self):
        cfg = R.default_cfg()
        for event,count in zip(cfg['ev'], [58,76,76,80]):
            event['n'] = count
        expected = R.build_no_gps_cure(cfg)
        result = self.run_js('const cfg='+json.dumps(cfg)+''';
comp=cfg.comp;ev=cfg.ev;
const M=buildNoGPSCure(cfg);
console.log(JSON.stringify({state:M.state,degenerate:M.degenerate,rms:M.rmsResid,
  limit:M.rmsTol,max:M.maxResid,shape:M.sG}));
''')
        self.assertEqual(result['state'], 'B')
        self.assertTrue(result['degenerate'])
        self.assertGreater(result['rms'], result['limit'])
        self.assertLess(result['rms'], 2.0)
        self.assertLess(result['max'], 3.0)
        self.assertTrue(0.16 < result['shape'] < 1.49)
        self.assertEqual(result['state'], expected['state'])
        self.assertAlmostEqual(result['rms'], expected['rmsResid'], places=9)
        self.assertAlmostEqual(result['limit'], expected['rmsTol'], places=12)

    def test_reach_warning_describes_full_followup_for_both_scenarios(self):
        start = self.html.index('function reachWarning(')
        warning = self.html[start:self.html.index('function render(){',start)]
        result = self.run_js(warning+'''
const models=['plateau','nogpscure'].map(kind=>({kind,cfg:{FINAL:80},MOBS:[60,72,78,80]}));
console.log(JSON.stringify({low:models.map(M=>reachWarning(M,0.45)),
  high:[0.7,1].map(r=>reachWarning(models[0],r)),
  custom:reachWarning({cfg:{FINAL:100},MOBS:[60,72,78,80]},0.4)}));
''')
        self.assertEqual(result['low'][0], result['low'][1])
        for note in result['low']:
            self.assertIn('45%', note)
            self.assertIn('full simulated follow-up', note)
            self.assertIn('already in the fitted history', note)
            self.assertIn('under-production of deaths', note)
            self.assertNotIn('stall', note)
            self.assertNotIn('by the', note)
        self.assertEqual(result['high'], ['', ''])
        self.assertNotIn('already', result['custom'])
        render = self.html[self.html.index('function render(){'):]
        note = render[render.index('let note='):render.index('const showFit =')]
        self.assertEqual(note.count('reachWarning(M,r.reach)'), 1)
        self.assertLess(note.index('reachWarning('), note.index('if(mode==='))
        self.assertNotIn('trigger may stall', self.html)

    def test_sweep_uses_the_weights_returned_by_the_model(self):
        start = self.html.index('function powerSweep(')
        source = self.html[start:self.html.index('\nconst lin=',start)]
        result = self.run_js(source+'''
function mc(){return {medHR:0.5,ps:0.7};}
const M={cfg:{},MT:[1,2,3,4],MOBS:[0,0,0,0],WT:[2,3,5,7],edRaw:(t,v)=>t*v};
console.log(JSON.stringify(powerSweep(M,'presp',[1],v=>[v],1)));
''')
        self.assertEqual(result[0]['E'], 2*1+3*4+5*9+7*16)


if __name__ == '__main__':
    unittest.main()

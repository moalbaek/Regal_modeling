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

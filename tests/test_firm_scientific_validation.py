"""Scientific checks use independent integrals, bounds and source invariance."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from firm_data import require_clean, sha256
from firm_science_reference import C,H,KB,Q,composition,enbw,gap,gap_dx,lorentzian_variance,pilot_rederivation,planck,simpson
from review_firm_seed import SOURCE,correct_detectivity,pilot_review
from build_firm_science_dev import development_items
from eval_firm_science import validate_benchmark,grade_quantity


class ScientificValidationTests(unittest.TestCase):
    def test_pilot_independent_rederivation_and_bounds(self):
        result=pilot_review()
        self.assertEqual(len(result['cases']),8)
        self.assertEqual(sum(len(x['quantities']) for x in result['cases']),22)
        r=pilot_rederivation()
        self.assertLess(r[0]['cutoff_150K'],9.4)
        self.assertAlmostEqual(r[2]['carrier_tau'],150e-6,places=10)
        self.assertLess(r[6]['absorption_20um'],.95)
        self.assertAlmostEqual(r[7]['enbw'],12.5)

    def test_all_detectivity_rows_individually_verified_and_others_unchanged(self):
        before=sha256(SOURCE); rows=require_clean(SOURCE);count=0
        for row in rows:
            original=copy.deepcopy(row)
            c=correct_detectivity(row,SOURCE)
            self.assertEqual(row,original)
            if c:
                count+=1
                p=c['metadata']['provenance']
                self.assertAlmostEqual(p['new_value']/(100*p['old_value']),1,delta=.02)
                self.assertLess(p['legacy_value_relative_error'],.02)
                self.assertEqual(c['messages'][:-1],row['raw']['messages'][:-1])
                self.assertIn('cm*sqrt(Hz)/W',c['messages'][-1]['content'])
        self.assertEqual(count,150);self.assertEqual(sha256(SOURCE),before)

    def test_planck_total_radiance_and_band_bound(self):
        t=350
        integral=simpson(lambda loglam:planck(math.exp(loglam),t)*math.exp(loglam),math.log(1e-8),math.log(.01),16384)
        sigma=2*math.pi**5*KB**4/(15*H**3*C*C)
        self.assertAlmostEqual(integral/(sigma*t**4/math.pi),1,places=8)
        band=simpson(lambda lam:planck(lam,t),8e-6,12e-6)
        self.assertLess(band,integral)

    def test_ENBW_independent_angle_integral(self):
        for n in [1,2,3,4]:
            numeric=simpson(lambda theta: math.cos(theta)**(2*n-2)/(2*math.pi*.02),0,math.pi/2)
            self.assertAlmostEqual(numeric,enbw(.02,n),places=10)

    def test_colored_filter_closed_antiderivative(self):
        a=2*math.pi*.004;A=1e-16;white=2e-18
        primitive=lambda f:A*math.log(f/math.sqrt(1+(a*f)**2))+white*math.atan(a*f)/a
        numeric=simpson(lambda f:(A/f+white)/(1+(a*f)**2),2,600)
        exact=primitive(600)-primitive(2)
        self.assertLess(abs(numeric/exact-1),1e-6)

    def test_lorentzian_integral_and_HSC_derivative(self):
        exact=lorentzian_variance(3e-18,.003,2,100)
        numeric=simpson(lambda f:3e-18/(1+(2*math.pi*f*.003)**2),2,100)
        self.assertAlmostEqual(numeric/exact,1,places=9)
        x=.24;t=100;dx=1e-6
        self.assertAlmostEqual((gap(x+dx,t)-gap(x-dx,t))/(2*dx),gap_dx(x,t),places=8)
        self.assertAlmostEqual(composition(9.4,85),.2288538863285693,places=12)

    def test_dev_release_quality_and_units(self):
        items=development_items();validate_benchmark(items)
        self.assertEqual(len(items),40)
        self.assertEqual(sum(len(x['grading']['quantities']) for x in items),106)
        for x in items:
            for q in x['grading']['quantities'].values():self.assertTrue(q['oracle'])
        # Independent midpoint quadrature of a finite Planck band.
        n=20000;lo=7.6e-6;hi=10.3e-6;step=(hi-lo)/n
        midpoint=math.fsum(planck(lo+(k+.5)*step,360) for k in range(n))*step
        self.assertAlmostEqual(midpoint/items[26]['grading']['quantities']['band_radiance']['value'],1,places=8)
        self.assertTrue(250<items[28]['grading']['quantities']['color_temperature']['value']<500)
        self.assertLess(items[30]['grading']['quantities']['external_QE']['value'],items[30]['grading']['quantities']['absorbed_incident_fraction']['value'])

    def test_added_unit_conversions_are_dimension_safe(self):
        gold={'value':1e21,'unit':'m^-3','rtol':1e-8,'atol':0}
        self.assertTrue(grade_quantity(gold,{'value':1e15,'unit':'cm^-3'})['numerical_correct'])
        self.assertFalse(grade_quantity(gold,{'value':1e21,'unit':'m^3/C'})['unit_correct'])
        self.assertTrue(grade_quantity({'value':.9,'unit':'m^2/(V*s)','rtol':1e-8,'atol':0},{'value':9000,'unit':'cm^2/(V*s)'})['numerical_correct'])
        self.assertTrue(grade_quantity({'value':42e-9,'unit':'V','rtol':1e-8,'atol':0},{'value':42,'unit':'nV'})['numerical_correct'])

    def test_reviewed_seed_has_disjoint_groups_and_no_erroneous_parents(self):
        root=ROOT/'data/processed/firm3_reviewed_seed_v3'
        manifest=json.loads((root/'manifest.json').read_text())
        groups={};ids={}
        for part in ['train','valid','test','quarantine']:
            rows=[r['raw'] for r in require_clean(root/(part+'.jsonl'))]
            groups[part]={r['metadata']['group_id'] for r in rows};ids[part]={r['id'] for r in rows}
            for r in rows:
                self.assertEqual(r['metadata']['partition'],part)
                if part!='quarantine':
                    self.assertIn(r['metadata']['provenance']['review_status'],['AI_analytically_reviewed','programmatically_corrected'])
                    self.assertNotEqual(r['metadata']['scientific_review']['review_kind'],'programmatically_corrected')
            for previous in groups:
                if previous!=part:self.assertFalse(groups[previous]&groups[part])
        self.assertEqual([len(ids[x]) for x in ['train','valid','test','quarantine']],[45,1,1,159])
        for output in manifest['outputs']:self.assertEqual(sha256(root/output['path']),output['sha256'])
        self.assertTrue(any(s['path']=='evals/firm_science_dev_v2.jsonl' for s in manifest['eval_sources']))

    def test_dev_v2_preserves_other_39_cases(self):
        old=development_items(1);new=development_items(2)
        differences=[i for i,(a,b) in enumerate(zip(old,new)) if a!=b]
        self.assertEqual(differences,[28])
        for version in [1,2]:
            actual=[r['raw'] for r in require_clean(ROOT/f'evals/firm_science_dev_v{version}.jsonl',evaluation=True)]
            self.assertEqual(actual,development_items(version))


    def test_compact_protocol_withholds_gold_and_cloud_models_rejected(self):
        case=development_items()[0]
        original=messages_for(case,'scientific-compact-v2')
        altered=copy.deepcopy(case)
        for q in altered['grading']['quantities'].values():q.update(value=98765,unit='madeup')
        altered['grading']['manual_rubric']=['SECRET GOLD']
        self.assertEqual(original,messages_for(altered,'scientific-compact-v2'))
        import io
        reply=io.BytesIO(json.dumps({'models':[{'name':'cloud:latest','digest':'x','remote_host':'ollama.com'}]}).encode())
        with patch('run_firm_baseline.urllib.request.urlopen',return_value=reply):
            with self.assertRaises(ValueError):ollama_metadata('http://127.0.0.1:11434','cloud:latest','x')

    def test_extracted_package_rejects_changed_payload_without_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);payload=root/'payload.txt';payload.write_text('trusted fixture')
            (root/'firm_gpu_package_manifest.json').write_text(json.dumps({'git_sha':'0'*40,'assets':[{'path':'payload.txt','sha256':sha256(payload)}]}))
            self.assertEqual(repository_revision(root),'0'*40)
            payload.write_text('changed fixture')
            with self.assertRaises(ValueError):repository_revision(root)

    def test_original_smoke_reviews_pin_all_32_rows(self):
        d=json.loads((ROOT/'data/reviews/firm3_manual_decisions_v1.json').read_text())
        old=ROOT/'data/processed/firm3_candidate_2026-10-05_v2/train.jsonl'
        self.assertEqual(sha256(old),d['historical_train_sha256'])
        rows=[x for x in d['decisions'] if x['smoke_index'] is not None]
        self.assertEqual(sorted(x['smoke_index'] for x in rows),list(range(1,33)))
        self.assertEqual(sum(x['status']=='exclude' for x in rows),6)
        self.assertEqual(sum(x['status']=='correct' for x in rows),1)


if __name__=='__main__':unittest.main()

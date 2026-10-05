"""Versioned scientific review/correction of selected seed sources; never overwrite.

Manual decisions are content-hash pinned. Uninspected examples remain unreviewed.
Only the strictly recognized legacy Jones family is repaired automatically.
"""
import argparse
import copy
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

from firm_data import NUMBER, calculation_checks, digest, load_evals, normalize, require_clean, sha256, template, write_json, write_jsonl
from firm_science_reference import detectivity, pilot_rederivation

SOURCE = Path('data/processed/firm_rewritten_large_sft.jsonl')
EXPERTS = sorted(Path('data/processed').glob('firm_v2*expert*sft.jsonl')) + sorted(Path('data/processed').glob('firm_v2_expert_hgcdte_deep_batch*.jsonl'))
NUM = r'([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)'
DSTAR = re.compile(r'A detector with area '+NUM+r' m\^2 and bandwidth '+NUM+r' Hz has NEP '+NUM+r' W\. Compute the specific detectivity \(D\*\)\.')


def example_id(row):
    return 'firm3_' + digest([row['input'].strip(), row['output'].strip()])


def correct_detectivity(row, source):
    match = DSTAR.fullmatch(normalize(row['input'], False))
    if not match:
        return None
    area, bandwidth, nep = map(float, match.groups())
    old = float(NUMBER.findall(normalize(row['output'], False))[-1])
    # Validate the actual old formula for EVERY row before applying a correction.
    legacy_expected = math.sqrt(area*bandwidth)/nep
    if 'Jones' not in row['output'] or not math.isclose(old, legacy_expected, rel_tol=.02):
        raise ValueError('Recognized prompt does not have the verified legacy error')
    value = detectivity(area, nep, bandwidth)
    corrected = copy.deepcopy(row['raw'])
    answer = (f'Treat the given NEP as RMS power over the stated effective noise bandwidth. '
              f'A = {area:.7g} m^2 = {area*1e4:.7g} cm^2. '
              'D* = sqrt(A_cm2 * Delta_f_Hz) / NEP_RMS_W. '
              f'Thus D* = {value:.9e} Jones (cm*sqrt(Hz)/W). '
              'This bandwidth normalization is comparable across bandwidths only for a matched, '
              'approximately white noise and responsivity convention; colored noise requires spectral characterization.')
    corrected['messages'][-1]['content'] = answer
    corrected['id'] = 'firm3_dstar_v1_' + digest([str(source), row['line'], row['input']])
    corrected['metadata']['family_id'] = 'legacy_Dstar_Jones_integrated'
    corrected['metadata']['provenance'] = {
        'kind': 'transformed', 'license': 'unknown', 'review_status': 'programmatically_corrected',
        'source': str(source), 'source_sha256': sha256(source), 'source_line': row['line'],
        'parent_id': example_id(row), 'transformation_version': 'Jones-SI-area-v1',
        'old_value': old, 'new_value': value, 'old_unit': 'Jones (incorrect label)', 'new_unit': 'Jones',
        'old_answer_sha256': digest(row['output']), 'old_answer': row['output'],
        'formula': 'sqrt((A_m2 * 1e4) * Delta_f_Hz) / NEP_RMS_W',
        'reason': 'Convert square metres to square centimetres before square root; recompute from printed inputs',
        'legacy_value_relative_error': abs(old-legacy_expected)/legacy_expected}
    return corrected


def pilot_review():
    path = Path('evals/firm_numeric_pilot_v1.jsonl')
    cases = [r['raw'] for r in require_clean(path, evaluation=True)]
    derived = pilot_rederivation()
    equations = ['Supplied Eg(x,T); Eg=hc/(q lambda); Newton root and derivative positive on bracket',
                 'NEP_ASD=e_n/Rv; 0.40 mm2=0.004 cm2; D*=sqrt(A_cm2)/NEP_ASD',
                 'Cascade magnitude squared multiplies; solve omega*tau after removing known RC magnitude',
                 'n SI=1e6*n_cm; mu SI=1e-4*mu_cm; R=L/(q*n*mu*W*t); drift transit=L/(mu*V/L)',
                 'Numerical integration in log(f) with Jacobian f; variance=int S df; RMS=sqrt(variance)',
                 'Decimal Planck in SI; B_per_um=B_per_m/1e6; P=B_per_m*dLambda*A*projectedOmega*transmission',
                 'Beer-Lambert: exp(-alpha*t); t95=ln(20)/alpha; cm to um factor 1e4',
                 'Gamma-function integral of one-sided unity-gain pole; ENBW=1/(4*tau), RMS=ASD*sqrt(ENBW)']
    checks = []
    for case, values, equation in zip(cases, derived, equations):
        result = []
        for name, value in values.items():
            gold = case['grading']['quantities'][name]
            relative = abs(value-gold['value'])/abs(gold['value'])
            if relative > 1e-10:
                raise ValueError('Independent pilot calculation disagrees: '+case['id']+' '+name)
            result.append({'name': name, 'independent_value': value, 'unit': gold['unit'],
                           'oracle_value': gold['value'], 'relative_difference': relative,
                           'rtol': gold['rtol'], 'atol': gold['atol']})
        checks.append({'id': case['id'], 'equation': equation, 'quantities': result,
                       'assumptions_checked': case['grading']['manual_rubric'],
                       'rubric_review': 'Appropriate bounded interpretation and controls; subjective scoring requires expert calibration',
                       'tolerance_review': '0.5% numerical tolerance permits rounding; not a bound on empirical-model uncertainty',
                       'status': 'independently_rederived_no_correction'})
    return {'schema_version': '1.0', 'reviewer': 'Codex independent analytic review; not human signoff',
            'review_version': 'pilot-independent-v1', 'source': str(path), 'sha256': sha256(path),
            'cases': checks, 'release_change': 'none; v1 preserved',
            'limitations': 'Public pilot; case 007 is a legacy-error regression. No claim of independent hidden-test generalization.'}


def build_review(manual_path):
    decisions = json.loads(manual_path.read_text())['decisions']
    manual = {d['example_id']: d for d in decisions}
    exclusions = json.loads(Path('data/manifests/firm3_review_exclusions.json').read_text())['exclusions']
    excluded = {(e['source'],e['line']):e for e in exclusions}
    audit = json.loads(Path('data/audits/firm3_2026-10-05/dataset_audit.json').read_text())
    flags = {(s['path'], f['line']): f for s in audit['sources']
             for f in s['metrics']['scientific_calculation_checks']['flagged']}
    rows, corrections, ledger, seen = [], [], [], set()
    for source in [SOURCE, *EXPERTS]:
        for row in require_clean(source):
            identifier = example_id(row)
            rec = {'example_id': identifier, 'source': str(source), 'source_sha256': sha256(source),
                   'source_line': row['line'], 'family_id': 'template_'+digest(template(row['input'])),
                   'status': 'needs_review', 'error_class': ['provenance'], 'review_notes': 'Not individually inspected; source license unknown',
                   'replacement_id': None, 'review_version': 'scientific-review-v1',
                   'review_kind': 'unreviewed', 'trainable': False}
            key = (str(source),row['line'])
            if key in flags:
                rec.update(error_class=['arithmetic','unit_conversion' if flags[key]['family']=='Dstar_Jones' else 'other'],
                           review_notes='Prior audit flag retained', audit_flag=flags[key])
            if re.search(r'(?:fraction|ionization|ionized)',row['input'],re.I) and re.search(r'activation energy',row['input'],re.I):
                rec.update(status='exclude', error_class=['missing_carrier_statistics','physical_assumption'],
                           review_notes='Ea and T do not fix Fermi level, degeneracy, concentration or charge neutrality')
            correction = correct_detectivity(row, source)
            if correction:
                corrections.append(correction)
                rec.update(status='correct', replacement_id=correction['id'], review_kind='programmatically_corrected',
                           error_class=['unit_conversion'], trainable=True,
                           family_id='legacy_Dstar_Jones_integrated', review_notes='Verified each old answer uses metre-based area then labels Jones')
            if identifier in manual:
                d = manual[identifier]
                if d['prompt_sha256'] != digest(row['input']) or d['answer_sha256'] != digest(row['output']):
                    raise ValueError('Manual decision content mismatch: '+identifier)
                rec.update({k:v for k,v in d.items() if k not in {'prompt_sha256','answer_sha256','smoke_index','replacement_answer'}})
                rec['review_kind'] = 'AI_analytic_review'
                if d.get('replacement_answer'):
                    rec['replacement_id'] = 'firm3_reviewed_v1_'+digest([identifier,d['replacement_answer']])
            if key in excluded:
                e = excluded[key]
                rec.update(status='exclude' if e.get('eval_id') else 'needs_review', trainable=False,
                           error_class=['eval_equivalent'] if e.get('eval_id') else ['measurement_model','physical_assumption'],
                           review_notes=e['reason'], prior_review=e)
            ledger.append(rec);seen.add(key)
            rows.append((row, rec, correction, manual.get(identifier)))
    # Preserve every flagged historical representation, not just the canonical SFT.
    for key,flag in sorted(flags.items()):
        if key in seen:
            continue
        source,line = key
        ledger.append({'example_id':'historical_'+digest([source,line]), 'source':source,
                       'source_line':line, 'source_sha256':sha256(source), 'family_id':flag['family'],
                       'status':'needs_review', 'error_class':['unit_conversion'] if flag['family']=='Dstar_Jones' else ['arithmetic'],
                       'review_notes':'Historical representation retained; not an independently eligible training source',
                       'replacement_id':None, 'review_version':'scientific-review-v1',
                       'review_kind':'prior_programmatic_flag','trainable':False,'audit_flag':flag})
    return rows, corrections, ledger


def reviewed_seed(manual_path, review_dir, out_dir, require_nonempty=False):
    """Reuse the grouped splitter, then restore explicit review/provenance states."""
    from split_firm_data import build_splits
    rows, _, _ = build_review(manual_path)
    eligible, unreviewed, excluded, review_exclusions = [], [], [], []
    for row, review, correction, manual in rows:
        if review['status'] not in {'accept','correct'} or not review['trainable']:
            ref={k:review[k] for k in ('example_id','source','source_line','source_sha256','status','review_notes')}
            (excluded if review['status']=='exclude' else unreviewed).append(ref)
            continue
        candidate=copy.deepcopy(row)
        if correction:
            candidate['raw']=correction
            candidate['messages']=correction['messages']
            candidate['output']=correction['messages'][-1]['content']
            candidate['metadata']=correction['metadata']
        elif manual and manual.get('replacement_answer'):
            candidate['output']=manual['replacement_answer']
            candidate['messages'][-1]['content']=manual['replacement_answer']
        candidate['metadata']['scientific_review']={k:review[k] for k in ('example_id','status','review_kind','review_notes','review_version','replacement_id')}
        candidate['metadata']['source_provenance']=candidate['metadata'].get('provenance',{
            'kind':'legacy_transformed','source':review['source'],'source_sha256':review['source_sha256'],
            'source_line':review['source_line'],'license':'unknown'})
        candidate['origins']=[{'path':review['source'],'sha256':review['source_sha256'],'line':review['source_line']}]
        # Benchmark numerical variants must not cross partitions. Reserve all
        # recognized legacy numerical families, even where the answer is correct.
        if calculation_checks([candidate])['count']:
            review_exclusions.append({'prompt_sha256':digest(normalize(candidate['input'])),
                'reason':'Conservative development reservation of recognized legacy quantitative template family'})
        eligible.append(candidate)
    splits,summary=build_splits(eligible,load_evals('evals'),seed=42,cap=3,review_exclusions=review_exclusions)
    adjustments=[]
    if require_nonempty:
        # Tiny reviewed releases can have an empty split under hash thresholds.
        # Deterministically move a WHOLE eligible group; never split a family.
        for target in ('test','valid'):
            if splits[target]:
                continue
            groups={r['metadata']['group_id'] for r in splits['train']}
            if len(groups)<2:
                raise ValueError('Insufficient independent groups for three nonempty partitions')
            chosen=min(groups,key=lambda g:digest([42,g]))
            moved=[r for r in splits['train'] if r['metadata']['group_id']==chosen]
            splits['train']=[r for r in splits['train'] if r['metadata']['group_id']!=chosen]
            for r in moved:r['metadata']['partition']=target
            splits[target]=moved
            adjustments.append({'group_id':chosen,'from':'train','to':target,'records':len(moved)})
    summary['groups_by_partition']={name:len({r['metadata']['group_id'] for r in records}) for name,records in splits.items()}
    summary['records_by_partition']={name:len(records) for name,records in splits.items()}
    for partition,records in splits.items():
        for record in records:
            meta=record['metadata'];review=meta['scientific_review']
            meta['provenance']={**meta.pop('source_provenance'),
                'review_status':'programmatically_corrected' if review['review_kind']=='programmatically_corrected' else 'AI_analytically_reviewed',
                'origins':meta['provenance']['origins']}
            meta['eligibility']='reviewed_science_candidate' if partition in {'train','valid','test'} else 'excluded'
            meta['rights_status']='unknown; explicit rights/provenance review before substantive release'
    if out_dir.exists():
        raise ValueError('Reviewed seed exists; choose a new version')
    out_dir.mkdir(parents=True)
    for name,records in {**splits,'excluded':excluded,'unreviewed':unreviewed}.items():
        write_jsonl(out_dir/(name+'.jsonl'),records)
    write_json(out_dir/'manifest.json',{'schema_version':'1.0','release':out_dir.name,
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'seed':42,'method':'existing hash-assigned connected groups; cap3; whole-family quarantine',
        'scientific_review':'Selected rows individually AI-reviewed or verified programmatic corrections; not human signoff',
        'rights_status':'unresolved; E2 release gate',
        'review_manifest':{'path':str(review_dir/'manifest.json'),'sha256':sha256(review_dir/'manifest.json')},
        'manual_decisions':{'path':str(manual_path),'sha256':sha256(manual_path)},
        'tool_sha256':{p.name:sha256(p) for p in [Path(__file__),Path('scripts/split_firm_data.py'),Path('scripts/firm_data.py')]},
        'sources':[{'path':str(p),'sha256':sha256(p)} for p in [SOURCE,*EXPERTS]],
        'eval_sources':[{'path':str(p),'sha256':sha256(p)} for p in sorted(Path('evals').glob('*.jsonl'))],
        'outputs':[{'path':p.name,'sha256':sha256(p),'records':len(p.read_text().splitlines())} for p in sorted(out_dir.glob('*.jsonl'))],
        'small_release_partition_adjustments':adjustments,
        'accounting':{'canonical_records':len(rows),'individually_or_programmatically_reviewed':len(eligible),
                      'unreviewed_or_other_flags':len(unreviewed),'excluded':len(excluded)},'summary':summary})
    print(json.dumps({**{name:len(v) for name,v in splits.items()},'excluded':len(excluded),'unreviewed':len(unreviewed)},sort_keys=True))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manual',type=Path,default=Path('data/reviews/firm3_manual_decisions_v1.json'))
    ap.add_argument('--out-dir',type=Path,default=Path('data/reviews/firm3_scientific_v1'))
    ap.add_argument('--build-seed', action='store_true')
    ap.add_argument('--seed-out-dir',type=Path,default=Path('data/processed/firm3_reviewed_seed_v1'))
    ap.add_argument('--require-nonempty',action='store_true',help='Move whole groups deterministically if a tiny release has empty valid/test')
    args=ap.parse_args()
    if args.build_seed:
        reviewed_seed(args.manual,args.out_dir,args.seed_out_dir,args.require_nonempty)
        return
    if args.out_dir.exists():
        ap.error('Review release exists; choose a new version')
    rows,corrections,ledger=build_review(args.manual)
    args.out_dir.mkdir(parents=True)
    write_jsonl(args.out_dir/'review_ledger.jsonl',ledger)
    write_jsonl(args.out_dir/'detectivity_corrected_v1.jsonl',corrections)
    write_json(args.out_dir/'pilot_review.json',pilot_review())
    write_json(args.out_dir/'manifest.json', {'schema_version':'1.0','review_version':'scientific-review-v1',
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'tool_sha256':{p.name:sha256(p) for p in [Path(__file__),Path('scripts/firm_science_reference.py')]},
        'manual_decisions':{'path':str(args.manual),'sha256':sha256(args.manual)},
        'source_records':len(rows),'canonical_status_counts':dict(Counter(r['status'] for _,r,_,_ in rows)),
        'outputs':[{'path':p.name,'sha256':sha256(p)} for p in sorted(args.out_dir.glob('*'))]})
    print('canonical',len(rows),'corrections',len(corrections),'ledger',len(ledger))


if __name__=='__main__':
    main()

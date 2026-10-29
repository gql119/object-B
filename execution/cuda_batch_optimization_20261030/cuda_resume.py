"""Candidate-only binding of the accepted 61-batch native CUDA resume proof.

The AST adapter changes only retained fixture selection and probe identity/arm.
Native Trainer, PGD update math, phase assertions and zero tolerances stay intact.
There is no local CPU stand-in for the real CUDA resume result.
"""
import argparse
import ast
import hashlib
from pathlib import Path
from bounded_common import ROOT,BASE_PGD_SHA,add_arguments,admission,fixture_view
from robustness_s4_at.common import atomic,sha,write_once

RESUME_SOURCE_SHA='57f4afb59dce5e7fef9af1fbebef7905c3103a4bf1d4117e1bc735c3a7816f3c'


def adapted_run_source():
    path=ROOT/'robustness_s4_at/validate_resume_phase.py'
    if sha(path)!=RESUME_SOURCE_SHA:raise ValueError('Accepted native resume proof source drift')
    tree=ast.parse(path.read_text(encoding='utf8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    counts=dict(resources=0,fixture=0,fixture_create=0,data=0,attack=0,identity=0,batch=0,phase=0,steps=0,updates=0,decay=0)
    class Binding(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                name=node.targets[0].id
                if name=='resources' and ast.unparse(node.value)=='resource_status()':
                    counts['resources']+=1;node.value=ast.Name(id='_window_resources',ctx=ast.Load());return node
                if name=='checked' and ast.unparse(node.value)=='audit_accepted_inputs(p)':
                    counts['fixture']+=1
                    return ast.parse("_fixture_data,_fixture_records=fixture_view(args,args.out/'retained_fixture',count=684 if args.batch==36 else None)").body[0]
            return self.generic_visit(node)
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='fixtures':
                if ast.unparse(node.value)!="fixtures(args.out / 'fixtures', checked, p, count=684)":raise ValueError('Unexpected original fixture call')
                counts['fixture_create']+=1;return ast.Pass()
            return self.generic_visit(node)
        def visit_Call(self,node):
            if ast.unparse(node)=="dict(last_opt_step=56, accumulate=2, batches_per_epoch=19)":
                counts['phase']+=1;return ast.Name(id='_expected_phase',ctx=ast.Load())
            if isinstance(node.func,ast.Name) and node.func.id=='data_view':
                if ast.unparse(node)!="data_view(firstroot, 'clean', fixture_root=args.out / 'fixtures/clean')":raise ValueError('Unexpected original resume data contract')
                counts['data']+=1
                return ast.parse("data_view(firstroot,args.arm,fixture_root=args.out/'retained_fixture')",mode='eval').body
            if isinstance(node.func,ast.Name) and node.func.id=='dict':
                for kw in node.keywords:
                    if kw.arg=='batch' and isinstance(kw.value,ast.Constant) and kw.value.value==36:
                        counts['batch']+=1;kw.value=ast.parse('args.batch',mode='eval').body
                        node.keywords.append(ast.keyword(arg='weight_decay',value=ast.Name(id='_nominal_decay',ctx=ast.Load())))
                        counts['decay']+=1;break
                for kw in node.keywords:
                    if kw.arg=='native_batch':kw.value=ast.parse('args.batch',mode='eval').body
                    if kw.arg=='first_next_batch_accumulates_without_attempt':kw.value=ast.parse('args.batch==36',mode='eval').body
            return self.generic_visit(node)
        def visit_List(self,node):
            if ast.unparse(node)=='[56, 58]':
                counts['steps']+=1;return ast.Name(id='_expected_steps',ctx=ast.Load())
            return self.generic_visit(node)
        def visit_Compare(self,node):
            if ast.unparse(node)=="resumed.at_holder['observer'].updates != 1":
                counts['updates']+=1
                return ast.parse("resumed.at_holder['observer'].updates != _expected_resumed_updates",mode='eval').body
            return self.generic_visit(node)
        def visit_Subscript(self,node):
            if ast.unparse(node)=="p['pgd']['4']":
                counts['attack']+=1
                return ast.parse("p['pgd'][str(args.epsilon)]",mode='eval').body
            return self.generic_visit(node)
        def visit_Constant(self,node):
            messages={
                'Probe did not reach the real safe post-warmup accumulation2 boundary':'Native checkpoint did not match the declared safe completed-epoch batch phase',
                'Uninterrupted actual native update phase was not [noattempt57,attempt58]':'Uninterrupted native update phase did not match the declared batch proposal',
                'Expected two resumed batches and one actual SGD':'Expected two resumed batches and the declared actual SGD count'}
            if isinstance(node.value,str) and node.value in messages:return ast.Constant(value=messages[node.value])
            if node.value=='CUDA_NATIVE_ACCUMULATE2_PHASE':
                counts['identity']+=1
                return ast.parse("'BOUNDED_CANDIDATE_PHASE_'+args.implementation+'_'+args.arm+'_'+str(args.epsilon)+'_B'+str(args.batch)",mode='eval').body
            return node
    function=Binding().visit(function);function.name='_bound_resume_run'
    if counts!=dict(resources=1,fixture=1,fixture_create=1,data=1,attack=2,identity=1,batch=1,phase=1,steps=2,updates=1,decay=1):raise ValueError('Resume adapter did not bind exactly the declared sites: '+str(counts))
    ast.fix_missing_locations(function)
    return ast.unparse(function)+'\n',counts


def run(args):
    if args.device!='0':raise ValueError('Production candidate resume proof requires actual CUDA; no CPU substitute')
    resources=admission(args,__file__)
    source,counts=adapted_run_source()
    import json
    proposal=Path(__file__).parent/f'PROPOSED_TEST_PROTOCOL_B{args.batch}.json'
    policy=json.loads(proposal.read_text())
    # Actual native phase after three complete fixture epochs, without forcing
    # accumulate/optimizer/AMP. Batch36 keeps the accepted 684-image proof.
    nb=19 if args.batch==36 else (720+args.batch-1)//args.batch
    expected_phase=dict(last_opt_step=3*nb-1,accumulate=2 if args.batch==36 else 1,batches_per_epoch=nb)
    expected_steps=[56,58] if args.batch==36 else [3*nb,3*nb+1]
    import robustness_s4_at.validate_resume_phase as native_proof
    import robustness_s4_at.trainer as trainer_module
    from candidate_scalar_readback import build_candidate
    candidate=None
    if args.implementation!='original':candidate,_=build_candidate(BASE_PGD_SHA,vectorize_counts=args.implementation=='03')
    scope=dict(vars(native_proof),_window_resources=resources,fixture_view=fixture_view,
        _nominal_decay=policy['train_args']['weight_decay'],_expected_phase=expected_phase,
        _expected_steps=expected_steps,_expected_resumed_updates=1 if args.batch==36 else 2)
    exec(compile(source,'<isolated retained-fixture candidate resume binding>','exec'),scope)
    original=trainer_module.PGDAttack
    try:
        if candidate is not None:trainer_module.PGDAttack=candidate
        scope['_bound_resume_run'](args)
    finally:trainer_module.PGDAttack=original
    result=json.loads((args.out/'smoke_summary.json').read_text())
    expected_batches=3*nb+4
    if result['state']!='passed' or result['actual_outer_batches']!=expected_batches or result['tolerance']!=0:
        raise ValueError('Current candidate did not pass the exact production resume proof')
    result.update(implementation=args.implementation,arm=args.arm,epsilon_pixels=args.epsilon,
        original_resume_proof_source_sha256=RESUME_SOURCE_SHA,entry_sha256=sha(__file__),AST_binding_sites=counts,
        adapted_probe_sha256=hashlib.sha256(source.encode()).hexdigest(),full_original_census_repeated=False,
        declared_batch_proposal_sha256=sha(proposal),actual_batch=args.batch,
        full_scientific_gate=False,formal_ready=False,throughput_measured=False)
    write_once(args.out/'candidate_resume_summary.json',result);atomic(args.out/'status.json',result)


if __name__=='__main__':
    q=argparse.ArgumentParser();add_arguments(q);q.add_argument('--batch',type=int,choices=(36,48,64,72),default=36);run(q.parse_args())

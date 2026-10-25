"""Targeted CPU arithmetic tests; not production batch/CUDA readiness."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import torch
from .cache import OnDemandSafeBGTripletCache
from .runtime import ROOT, now, sha, write_once

REFERENCE = '34fee63a232d489ae54d61eb37a42483091c40d4'


def run(output):
    torch.set_num_threads(1)
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show', REFERENCE+':coco_s4/cache.py'])
    namespace = {}; exec(compile(source, 'retained_cache_reference.py', 'exec'), namespace)
    results = []
    for dtype in (torch.float32, torch.float64):
        generator = torch.Generator().manual_seed(157)
        bases = {c:torch.linalg.qr(torch.randn(24, 4, dtype=dtype, generator=generator)).Q for c in range(1,20)}
        old = namespace['OnDemandSafeBGTripletCache'](bases, max_entries=3)
        new = OnDemandSafeBGTripletCache(bases, max_entries=3)
        assert old.sha256 == new.sha256
        for order in ((1,2,3),(3,2,1),(4,5,6),(19,1,2)):
            for before, after in zip(old.lookup(order),new.lookup(order)):
                assert torch.equal(before,after), (dtype,order)
        results.append(dict(dtype=str(dtype),VOC24_all_cache_outputs_and_hash_exact=True))
    dynamic = []
    for dimension in (37,39,41):
        generator = torch.Generator().manual_seed(193)
        bases = {c:torch.linalg.qr(torch.randn(dimension,12,generator=generator)).Q for c in range(1,80)}
        cache = OnDemandSafeBGTripletCache(bases,max_entries=3)
        maximum = 0.; seen = set()
        for c in range(1,80):
            order = (c,c%79+1,(c+1)%79+1);seen.update(order)
            q,s,gram,rank,energy = cache.lookup(order)
            concatenated = torch.cat([bases[k] for k in order],1)
            u,values,_ = torch.linalg.svd(concatenated.double(),full_matrices=False)
            assert torch.equal(q,u[:,:8].float())
            assert torch.equal(s,values[:16].float())
            assert q.dtype==torch.float32 and q.shape==(dimension,8) and rank==8 and gram<=1e-5
            assert len(cache.entries)<=3
            maximum=max(maximum,float(gram))
        assert seen==set(range(1,80))
        memory=cache.memory_report();assert memory['entry_bytes']<=memory['entry_upper_bound_bytes']
        dynamic.append(dict(actual_dimension=dimension,ordered_triplets=79,all79_classes_covered=True,
            same_ordered_float64_SVD_then_cast_exact=True,rank=8,maximum_Gram=maximum,
            thresholds_unchanged=True,cache_memory=memory))
    assert not torch.cuda.is_initialized()
    report=dict(state='passed',UTC=now(),reference=REFERENCE,reference_source_sha256=hashlib.sha256(source).hexdigest(),
        cache_source_sha256=sha(ROOT/'coco_s4/cache.py'),VOC24=results,dynamic_dimensions=dynamic,
        synthetic_fixture_only=True,model_forwards=0,GPU_initialized=False,
        learned_Teacher_batch16_CPU='OPEN',current_source_batch16_CUDA='OPEN',formal_ready=False)
    write_once(Path(output)/'report.json',report)
    print(__import__('json').dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);run(p.parse_args().output)

"""Small durable records. No destructive operations or external communication."""
import hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'robustness_s4_extended/configs/QUEUE.json'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(x,f,indent=2,allow_nan=False,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def commit():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def verify_source(config):
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('Tracked source dirty')
    for rel,digest in config['source_files'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Source differs: '+rel)
    validate_config(config)

def validate_config(c):
    if c['order']!=['cutout','cutmix','mixup','op'] or c['OP_PROTOCOL_APPROVED'] is not False:raise ValueError('Queue or OP approval changed')
    a=c['baseline']['victim_protocol']['train_args']
    if (a['epochs'],a['batch'],a['workers'],a['seed'],a['amp'],a['optimizer'],a['imgsz'],a['close_mosaic'],a['pretrained'])!=(200,36,16,0,True,'SGD',640,100,False):raise ValueError('Frozen victim protocol differs')
    if c['defenses']['op']['enabled'] is not False or c['defenses']['op']['OP_PROTOCOL_APPROVED'] is not False:raise ValueError('OP prototype quarantine violated')
    for name in ['cutout','cutmix','mixup']:
        x=c['defenses'][name]
        if x['name']!=name or x['probability']!=1.:raise ValueError('Unreviewed defense parameters')
        if load(ROOT/'robustness_s4_extended/configs'/(name.upper()+'_CONFIG.json'))!=x:raise ValueError('Config copies differ')
    x=c['defenses']['cutout'];y=c['defenses']['cutmix'];z=c['defenses']['mixup']
    if (x['n_holes'],x['area_fraction'],x['max_gt_occlusion'],x['max_attempts'],x['fill_value'])!=(1,[.25,.25],.5,8,0):raise ValueError('Cutout protocol drift')
    if (y['beta'],y['num_areas'],y['donor_min_visible'],y['min_pixels'])!=(1.,3,.5,2.):raise ValueError('CutMix protocol drift')
    if z['beta']!=[32.,32.]:raise ValueError('MixUp must use pinned native beta32/32')
    if 'selection' in c:
        from .selection import validate_rule
        p=ROOT/c['selection']['rule_path'];validate_rule(load(p))
        if sha(p)!=c['selection']['rule_sha256']:raise ValueError('Selection rule bytes changed')

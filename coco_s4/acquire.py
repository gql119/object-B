"""Public COCO bytes -> bounded local tar batches -> RRCTL static-data store.

No image re-encoding, GPU, training, environment changes, or dataset sampling.
Local artifacts and failed attempts are retained. A persistent source-bound
manifest, local process lock and absent-or-identical server writes prevent
overwrites. This is data acquisition, never a GPU scheduler.
"""
import argparse,concurrent.futures,datetime,hashlib,io,json,msvcrt,os,shutil,sys,tarfile,time,urllib.request
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
LOCAL=Path('F:/autodl-tmp - 副本/ue_project/remote_artifacts/coco_minitrain_acquisition_20261023')
REMOTE='/data/guoqinglong/datasets/COCO_MINITRAIN_25K_S4_20261023'
PYTHON='/project/guoqinglong/miniconda3/envs/cfbdm/bin/python'
EXPECTED_SOURCES={'train':'5682389c3bb0a356912771811f7836659cc6de5a588413752922ccb94a6df1a8','val':'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'}

def sha(data):return hashlib.sha256(data).hexdigest()
def atomic(path,value):
    p=path.with_suffix('.writing');p.write_text(json.dumps(value,indent=2));os.replace(p,path)

EXTRACT=r'''
from pathlib import Path
import hashlib,io,json,os,shutil,tarfile,time
from PIL import Image
root=Path(P['root']);assert root.stat().st_uid==os.getuid()
assert shutil.disk_usage(root).free>=150*2**30
mem={s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')};assert mem['MemAvailable']>=32*2**20
archive=Path(P['archive']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==P['archive_sha256']
if P['resource_sample']:
 waits=[]
 def cpu():return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]))
 for _ in range(3):
  a=cpu();time.sleep(1);b=cpu();delta=[y-x for x,y in zip(a,b)];waits.append(100*delta[4]/max(1,sum(delta[:8])))
 assert not all(v>=20 for v in waits),'WAITING_SUSTAINED_IO_RESOURCES'
records={x['relative']:x for x in P['records']};written=[]
started=time.monotonic();total_bytes=0
with tarfile.open(archive,'r:') as t:
 assert len(t.getmembers())==len(records)
 for m in t.getmembers():
  assert m.isfile() and m.name in records and len(Path(m.name).parts)==2 and Path(m.name).parts[0] in ['train','val'] and m.size<=32*2**20
  r=records[m.name];data=t.extractfile(m).read();assert hashlib.sha256(data).hexdigest()==r['sha256']
  with Image.open(io.BytesIO(data)) as im:assert im.size==tuple(r['dimensions']);im.verify()
  dest=root/'images'/m.name;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.stat().st_uid==os.getuid() and hashlib.sha256(dest.read_bytes()).hexdigest()==r['sha256']
  else:
   with dest.open('xb') as f:f.write(data)
  total_bytes+=len(data);delay=total_bytes/(2*2**20)-(time.monotonic()-started)
  if delay>0:time.sleep(delay)
  written.append(m.name)
receipt={'state':'ACQUIRED_VERIFIED','batch':P['batch'],'source_sha256':P['source_sha256'],'archive_sha256':P['archive_sha256'],'records':P['records'],'images_verified':len(written),'method':'raw official HTTP JPEG bytes; no re-encoding','no_GPU_or_training':True}
out=root/'metadata'/(P['batch']+'.json')
if out.exists():assert json.loads(out.read_text())==receipt
else:out.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--batches',type=int,default=1);args=p.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(LOCAL).free<20*2**30:raise ValueError('Need20GiB local free space, reserve10GiB throughout; no cleanup permitted')
    lock=(LOCAL/'acquisition.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    plan=[];identities={}
    for split,name in [('train','instances_minitrain2017.json'),('val','instances_val2017.json')]:
        src=ROOT/'execution/private_annotations'/name;data=src.read_bytes();identities[split]=sha(data);assert identities[split]==EXPECTED_SOURCES[split];obj=json.loads(data)
        assert len(obj['images'])==(25000 if split=='train' else 5000)
        for im in sorted(obj['images'],key=lambda x:x['id']):
            relative=f"{split}/{im['id']:012d}.jpg";url=f"https://s3.amazonaws.com/images.cocodataset.org/{split}2017/{im['id']:012d}.jpg"
            assert im['file_name']==Path(relative).name
            plan.append({'relative':relative,'url':url,'dimensions':[im['width'],im['height']]})
    source_hash=sha(json.dumps(identities,sort_keys=True).encode());status_file=LOCAL/'STATUS.json'
    state=json.loads(status_file.read_text()) if status_file.exists() else {'source_sha256':source_hash,'completed_batches':{},'images_verified':0,'state':'PREPARING'}
    assert state['source_sha256']==source_hash
    state.update(pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(Path(__file__).read_bytes()),total_images=30000)
    sys.path.insert(0,'F:/autodl-tmp - 副本/ue_project/.agents/harness/remote/rrctl/src')
    from remote_run_control.profiles import ProfileStore
    from remote_run_control.transport import transport_for
    t=transport_for(ProfileStore(BASE/'s4_robustness_evaluation_20261019/execution/profiles.json').load('school-s1'))
    def remote(code,payload):
        text='import json\nP=json.loads('+repr(json.dumps(payload))+')\n'+code
        r=t.run([PYTHON,'-B','-'],input_data=text.encode())
        if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
        return json.loads(r.stdout)
    remote("from pathlib import Path; import json; r=Path(P['root']);(r/'archives').mkdir(exist_ok=True);print(json.dumps({'root_exists':r.exists()}))",{'root':REMOTE})
    completed_now=0
    def fetch(entry):
        errors=[]
        for retry in range(3):
            try:
                with urllib.request.urlopen(entry['url'],timeout=45) as response:
                    if response.headers.get_content_type() not in ['image/jpeg','application/octet-stream']:raise ValueError('Official image endpoint returned non-image content')
                    payload=response.read(32*2**20+1)
                if len(payload)>32*2**20:raise ValueError('Oversized image')
                with Image.open(io.BytesIO(payload)) as im:
                    if im.format!='JPEG' or im.size!=tuple(entry['dimensions']):raise ValueError('Official image format/dimensions differ')
                    im.verify()
                return entry,payload,errors
            except Exception as e:
                errors.append({'image':entry['relative'],'attempt':retry+1,'error':str(e)})
                if retry==2:raise RuntimeError(json.dumps(errors)) from e
    try:
        for i,start in enumerate(range(0,len(plan),32)):
            if shutil.disk_usage(LOCAL).free<10*2**30:raise ValueError('Local disk reserve10GiB reached; retain artifacts')
            key=f'official_batch_{i:04d}'
            if key in state['completed_batches']:continue
            if completed_now>=args.batches:break
            state.update(state='DOWNLOADING',active_batch=key);atomic(status_file,state)
            attempt=1
            while (LOCAL/f'{key}_attempt{attempt}.tar').exists():attempt+=1
            archive=LOCAL/f'{key}_attempt{attempt}.tar';records=[]
            with tarfile.open(archive,'x:') as tar,concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for entry,payload,errors in pool.map(fetch,plan[start:start+32]):
                    state.setdefault('retained_failures',[]).extend(errors)
                    record=dict(entry,sha256=sha(payload),bytes=len(payload));records.append(record)
                    member=tarfile.TarInfo(entry['relative']);member.size=len(payload);member.mode=0o600;tar.addfile(member,io.BytesIO(payload))
                    state['downloaded_in_active_batch']=len(records);atomic(status_file,state)
            state.update(state='UPLOADING');atomic(status_file,state)
            remote_archive=REMOTE+'/archives/'+archive.name;archive_sha=sha(archive.read_bytes());t.upload(archive,remote_archive,archive_sha)
            receipt=remote(EXTRACT,{'root':REMOTE,'archive':remote_archive,'archive_sha256':archive_sha,'batch':key,'records':records,'source_sha256':source_hash,'resource_sample':i%16==0})
            assert receipt['images_verified']==len(records)
            state['completed_batches'][key]={'archive':str(archive),'archive_sha256':archive_sha,'records':records}
            state['images_verified']+=len(records);state.update(state='BATCH_VERIFIED',active_batch=None);atomic(status_file,state);completed_now+=1
            print(json.dumps({'batch':key,'images_verified_total':state['images_verified'],'no_GPU':True}),flush=True)
        state['state']='ALL_IMAGES_ACQUIRED_PENDING_FULL_DATA_AUDIT' if state['images_verified']==30000 else 'PARTIAL_ACQUISITION'
        atomic(status_file,state);print(json.dumps({'state':state['state'],'images_verified':state['images_verified']}))
    except Exception as e:
        state.update(state='BLOCKED_ACQUISITION',error=str(e));atomic(status_file,state);raise
    finally:lock.close()
if __name__=='__main__':main()

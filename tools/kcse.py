#!/usr/bin/env python3
"""Relocatable entry point. Frozen scientific sources are never rewritten."""
from pathlib import Path
import argparse, hashlib, json, os, re, runpy, shutil, subprocess, sys, tempfile

ROOT = Path(__file__).resolve().parents[1]
IMAGES = {
 'generation': ('kcse-planner-generation_v3--ae52f8a4d0e8.tar.zst', 'sha256:6bc378e182e3cc95edffff6589166e63558423f07f017b89b2d1b7f32340e0ad'),
 'evaluation': ('kcse-bugsinpy_revision-v2-py38--2ff43c4b906e.tar.zst', 'sha256:eb2e457d0685eb31003fa6e1ff6a6fac6ce3230765ec7bf494747fd9111b71a0'),
}

def sha(path):
 h = hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
 return h.hexdigest()

def command(*args, **kwargs):
 return subprocess.run([str(a) for a in args], check=True, **kwargs)

def check_manifest(directory):
 for line in (directory/'SHA256SUMS.txt').read_text().splitlines():
  digest, name = line.split(None, 1)
  path = directory/name.lstrip(' *')
  if sha(path) != digest: raise RuntimeError(f'checksum mismatch: {path}')

def verify():
 entries = json.loads((ROOT/'inventory/SHA256.json').read_text())
 for i, (name, digest) in enumerate(entries.items()):
  path=ROOT/name
  if path.is_symlink() or not path.is_file() or sha(path)!=digest:
   raise RuntimeError(f'file missing, linked or altered: {name}')
 print(f'PASS: {len(entries)} independent files checked')
 for directory in [ROOT/'generation/bundle', *sorted((ROOT/'evaluation').glob('*/bundle'))]:
  check_manifest(directory)
 audit = json.loads((ROOT/'evidence/final_audit/MANIFEST.json').read_text())
 for row in audit['records']:
  record_path=ROOT/'evaluation'/row['evaluation_record']
  assert sha(record_path)==row['record_sha256']
  record=json.loads(record_path.read_text())[str(row['bug_id'])]
  candidate=ROOT/f"evaluation/kcse_full_evaluation_v{row['evaluation_version']}/bundle/candidates/{row['arm']}_seed{row['seed']}/{row['bug_id']}.json"
  assert candidate.is_file()
  if record.get('result_file_sha256'): assert sha(candidate)==record['result_file_sha256']
 for row in audit['generation_logs']:
  assert sha(ROOT/'generation/results'/row['run_dir']/'model-inputs.json')==row['model_inputs_sha256']
 assert sha(ROOT/'models/planner/best_model_state_dict.pt')=='b02f95aa1fcb24b19f5a5e038e47976125ce7979a686821810465c3f4ee2f730'
 expected=json.loads((ROOT/'generation/bundle/assets/model-manifest.json').read_text())
 for name,digest in expected.items(): assert sha(ROOT/'models/starcoder2-7b'/name)==digest, name
 assert sha(ROOT/'models/e5-small-v2/model.safetensors')=='45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1'
 check_manifest(ROOT/'training')
 assert sha(ROOT/'training/assets/random_split.json')=='833df6f0749fbe8da77ef7a32c936fbdf8a72005f9418cc69c7a96dbb06ec09b'
 assert sha(ROOT/'training/repo/scripts/train_planner_archived_rng_v4.py')=='032cce9b01a3e6b0d861b42d82e496296244b0aac1eea471dfaba19542672ce8'
 print('PASS: frozen bundles, 258 evaluation/candidate references, 258 generation inputs, model identity and training split')

def aggregate(output,k=1):
 # Loading under a non-main name retains original code bytes and metric behavior.
 sys.path.insert(0,str(ROOT/'generation/bundle/repo'))
 path=ROOT/'evaluation/kcse_full_evaluation_v5/analysis/aggregate_full_results.py'
 module=runpy.run_path(str(path),run_name='kcse_frozen_aggregate')
 namespace=module['main'].__globals__
 namespace.update(GEN_ROOT=ROOT/'generation', EVAL_ROOTS=[ROOT/f'evaluation/kcse_full_evaluation_v{v}' for v in range(1,6)])
 sys.argv=[str(path),'--k',str(k),'--output',str(output)]
 module['main']()
 expected=json.loads((path.parent/'ACCOUNTING.json').read_text())
 actual=json.loads(output.read_text())
 keys=['identity','stage_coverage']
 if k==1: keys+=['overall','per_arm','per_group','rows','unresolved_reasons']
 for key in keys:
  if actual[key]!=expected[key]: raise RuntimeError(f'accounting mismatch: {key}')
 print('PASS: aggregate matches preserved ACCOUNTING.json' if k==1 else f'PASS: prefix@{k} computed with frozen aggregator and preserved denominator')

def load_image(kind):
 if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Submit a real Slurm job first')
 filename,image=IMAGES[kind]
 if subprocess.run(['/usr/local/bin/cehub-docker','exec','image','inspect',image],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
  return image
 ref=f'kcse-portable:{kind}-20261026'
 # cehub-docker load uses the authorized user's permanent image store.
 # Separate registration avoids changing the user's historical archives/symlinks.
 name=re.sub(r'[^A-Za-z0-9_.-]+','_',ref).strip('._-')[:48]+'--'+hashlib.sha256(ref.encode()).hexdigest()[:12]+'.tar.zst'
 store=Path.home()/'.cehub/docker-images';store.mkdir(parents=True,exist_ok=True)
 target=store/name; source=ROOT/'images'/filename
 if target.is_symlink(): raise RuntimeError(f'remove conflicting registration symlink: {target}')
 if not target.exists():
  temporary=target.with_name(target.name+f'.{os.getpid()}.partial')
  shutil.copy2(source,temporary);os.replace(temporary,target)
 if sha(target)!=sha(source): raise RuntimeError('registered image differs from package')
 target.with_suffix('.json').write_text(json.dumps({'image':ref})+'\n')
 command('/usr/local/bin/cehub-docker','images',stdout=subprocess.DEVNULL)
 present=subprocess.run(['/usr/local/bin/cehub-docker','exec','image','inspect',image],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
 if not present: command('/usr/local/bin/cehub-docker','load',ref)
 command('/usr/local/bin/cehub-docker','exec','image','inspect',image,stdout=subprocess.DEVNULL)
 return image

def container(kind, bundle, output, inside, gpu=False, training=False):
 image=load_image(kind)
 workspace=tempfile.TemporaryDirectory(prefix='kcse-workspace-') if kind=='evaluation' else None
 output.mkdir(parents=True,exist_ok=False)
 args=['/usr/local/bin/cehub-docker','exec','run','--rm','--network=none','--entrypoint','/bin/bash',
       '-e',f'SLURM_JOB_ID={os.environ["SLURM_JOB_ID"]}',
       '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
       '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','OMP_NUM_THREADS=4',
       '--mount',f'type=bind,src={bundle},dst=/bundle,readonly',
       '--mount',f'type=bind,src={output},dst=/output','--workdir','/bundle/repo']
 if training:
  args+=['--mount',f'type=bind,src={ROOT/"tools"},dst=/portable,readonly',
         '-e','KCSE_PACKAGE_IMAGE_SHA256='+sha(ROOT/'images'/IMAGES['generation'][0])]
 if kind=='evaluation':
  args+=['--mount',f'type=bind,src={workspace.name},dst=/output/workspaces']
  args+=['--mount',f'type=bind,src={ROOT/"offline"},dst=/offline,readonly',
         '--mount',f'type=bind,src={ROOT/"tools"},dst=/portable,readonly']
  inside=['/portable/evaluate_offline.sh',*inside]
 if gpu:
  args+=['--device=nvidia.com/gpu=all','--shm-size=1g']
  for local,remote in [('models/starcoder2-7b','/models/starcoder2-7b'),('models/e5-small-v2','/models/e5-small-v2'),('models/planner','/planner')]:
   args+=['--mount',f'type=bind,src={ROOT/local},dst={remote},readonly']
  for name in ['KCSE_ARM','KCSE_BUG_ID','KCSE_SEED']:
   if name in os.environ: args+=['-e',f'{name}={os.environ[name]}']
 try:
  with (output/'stdout.txt').open('w') as out,(output/'stderr.txt').open('w') as err:
   command(*args,image,*inside,stdout=out,stderr=err)
 finally:
  if workspace is not None: workspace.cleanup()


def generate(index, output):
 tasks=json.loads((ROOT/'generation/bundle/contract/tasks.json').read_text())
 task=tasks[index]
 for key,field in [('KCSE_ARM','arm'),('KCSE_BUG_ID','bug_id'),('KCSE_SEED','seed')]: os.environ[key]=str(task[field])
 container('generation',ROOT/'generation/bundle',output,
           ['/bundle/repo/scripts/run_original_method_pilot_inside_20260920.sh'],gpu=True)
 assert (output/'GENERATION_COMPLETED.json').is_file()
 print(f'Generation complete: {output}')

def evaluate(index, output, generated=None):
 tasks=json.loads((ROOT/'generation/bundle/contract/tasks.json').read_text());task=tasks[index]
 rows=json.loads((ROOT/'evidence/final_audit/MANIFEST.json').read_text())['records']
 row=next(r for r in rows if str(r['bug_id'])==str(task['bug_id']) and r['arm']==task['arm'] and r['seed']==task['seed'])
 version=row['evaluation_version'];bundle=ROOT/f'evaluation/kcse_full_evaluation_v{version}/bundle'
 protocol=json.loads((bundle/'protocol.json').read_text())
 run=f"{task['arm']}_seed{task['seed']}"
 planned=next(t for t in protocol['tasks'] if t['run']==run and str(task['bug_id']) in [str(x) for x in t['ids']])
 script='candidate_revalidation_abi_20260921.sh' if version==5 else 'candidate_revalidation_inside_20260919.sh'
 provenance=None
 with tempfile.TemporaryDirectory(prefix='kcse-new-candidates-') as temporary:
  if generated is not None:
   source=generated/'result.json' if generated.is_dir() else generated
   if not (source.parent/'GENERATION_COMPLETED.json').is_file(): raise RuntimeError('generation completion record missing')
   result=json.loads(source.read_text())
   if (str(result['bug_id']),result['arm'],int(result['seed']))!=(str(task['bug_id']),task['arm'],task['seed']):
    raise RuntimeError('generated result does not match --task identity')
   original_manifest=sha(bundle/'SHA256SUMS.txt')
   stage=Path(temporary)/'bundle'
   shutil.copytree(bundle,stage)
   shutil.copy2(source,stage/f"candidates/{run}/{task['bug_id']}.json")
   provenance={'kind':'new_candidate_evaluation','source_result_sha256':sha(source),
               'original_evaluation_bundle_manifest_sha256':original_manifest,'task':task}
   (stage/'candidate-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
   lines=[sha(f)+'  '+str(f.relative_to(stage)) for f in sorted(stage.rglob('*'))
          if f.is_file() and f.name!='SHA256SUMS.txt' and '__pycache__' not in f.parts]
   (stage/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
   bundle=stage
  container('evaluation',bundle,output,[f'/bundle/repo/scripts/{script}',run,str(task['bug_id']),str(planned['compile_timeout'])])
 if provenance is not None: (output/'new-candidate-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 record=json.loads((output/'evaluation.json').read_text())[str(task['bug_id'])]
 if record.get('status')!='Success': raise RuntimeError(f'evaluation unresolved: {record.get("status")}')
 print(f'Evaluation complete: {output}')

def train(output,mode):
 container('generation',ROOT/'training',output,['/portable/train_inside.sh',mode],gpu=mode=='primary',training=True)
 print(f'Training {mode} complete: {output}')

def submit(mode,index,output,training_mode='primary',generated=None):
 gpu=mode=='generate' or (mode=='train' and training_mode=='primary')
 # One submitted job, one worker node; no hidden multi-node array.
 args=['/usr/local/bin/cehub-slurm','sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4',
       '--mem='+('32G' if gpu else '8G'),'--time='+('24:00:00' if mode=='train' and training_mode=='primary' else '02:00:00'),
       '--job-name=kcse-portable-'+mode,'--output='+str(output)+'.slurm.log']
 if gpu: args+=['--gres=gpu:1']
 import shlex
 arguments=['python3',str(Path(__file__).resolve()),mode,'--task',str(index),'--mode',training_mode,'--output',str(output)]
 if generated is not None: arguments+=['--generated',str(generated)]
 line=shlex.join(arguments)
 command(*args,'--wrap',line)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('action',choices=['verify','aggregate','generate','evaluate','submit-generate','submit-evaluate','train','submit-train'])
 parser.add_argument('--task',type=int,default=0)
 parser.add_argument('--k',type=int,choices=[1,5,10],default=1)
 parser.add_argument('--mode',choices=['smoke','primary'],default='primary')
 parser.add_argument('--output',type=Path)
 parser.add_argument('--generated',type=Path,help='new generation output directory or result.json; evaluate actions only')
 args=parser.parse_args()
 if args.generated is not None:
  if args.action not in {'evaluate','submit-evaluate'}: parser.error('--generated requires evaluate or submit-evaluate')
  args.generated=args.generated.resolve()
 if not 0<=args.task<258: parser.error('--task must be between 0 and 257')
 output=(args.output or ROOT/'outputs'/args.action/str(args.task)).resolve()
 output.parent.mkdir(parents=True,exist_ok=True)
 if args.action=='verify':verify()
 elif args.action=='aggregate':aggregate(output,args.k)
 elif args.action=='generate':generate(args.task,output)
 elif args.action=='evaluate':evaluate(args.task,output,args.generated)
 elif args.action=='train':train(output,args.mode)
 else:submit(args.action.removeprefix('submit-'),args.task,output,args.mode,args.generated)

if __name__=='__main__': main()

"""TrustMR experiments: VSQ vector correctness, parameter screening, and holdout evaluation."""
import os,sys,time,csv,json,pickle,copy,struct,hashlib,shutil,multiprocessing as mp,gc
from pathlib import Path
import numpy as np
from core import *
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1];OUT=ROOT/'results/vsq_vector_v1';GT=ROOT/'results/evaluation_manifest_v1'
CONFIGS={'V1':(4,64),'V2':(8,32),'V3':(16,16),'V4':(32,8)}
META=list(csv.DictReader((GT/'query_ids.csv').open()));QID={(r['modality'],int(r['query_index'])):r['query_id'] for r in META}
QS={m:np.load(GT/f'{m}_queries.npy',mmap_mode='r') for m in ('image','text')};GTS={m:np.load(GT/f'{m}_gt_ids.npy',mmap_mode='r') for m in ('image','text')}
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def table(name,rows,fields=None):
 fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def base_data():
 points=np.load(GT/'corpus_vectors.npy',mmap_mode='r');shared=OUT/'shared';shared.mkdir(exist_ok=True)
 pp=shared/'projection.npy'
 if not pp.exists():shutil.copyfile(ROOT/'results/vgann_final_screen_v1/trustmr_projection.npy',pp)
 proj=np.load(pp);t=time.perf_counter()
 if not (shared/'signatures.npy').exists():
  print('SIGNATURES BUILD',flush=True);sigs=np.packbits(np.einsum('ij,kj->ik',points,proj,dtype=np.float64,optimize=False)>=0,axis=1,bitorder='little');np.save(shared/'signatures.npy',sigs)
  hashes=np.asarray([np.frombuffer(vh(v),dtype='u1') for v in points]);np.save(shared/'vector_hashes.npy',hashes)
  (shared/'record_commitments.bin').write_bytes(b''.join(rh(i,sigs[i],hashes[i]) for i in range(len(points))))
  save('signature_build.json',{'seconds':time.perf_counter()-t,'projection_source':'results/vgann_final_screen_v1/trustmr_projection.npy','normalization_repeated':False})
 sigs=np.load(shared/'signatures.npy',mmap_mode='r');hashes=np.load(shared/'vector_hashes.npy',mmap_mode='r')
 for i in range(40):assert signature(points[i],proj)==bytes(sigs[i])
 return points,sigs,hashes,proj

def persist(ix,name):
 folder=OUT/name;folder.mkdir(exist_ok=True);offset=0
 with (folder/'bucket_directory.bin').open('wb') as d,(folder/'members.bin').open('wb') as f:
  for key,group in zip(ix.tree.keys,ix.tree.members):
   d.write(struct.pack('<HQQI',key[0],key[1],offset,len(group)));f.write(group.astype('<u4',copy=False).tobytes());offset+=len(group)
 (folder/'member_digests.bin').write_bytes(b''.join(ix.tree.digests));(folder/'tree.bin').write_bytes(b''.join(h for layer in ix.tree.levels for h in layer))
 a=anchor(ix);save(name+'/anchor.json',{**a,'root':a['root'].hex()})
 shared=OUT/'shared';part={key:(shared/file).stat().st_size for key,file in [('projection_bytes','projection.npy'),('signatures_bytes','signatures.npy'),('vector_hashes_bytes','vector_hashes.npy'),('record_commitment_cache_bytes','record_commitments.bin')]}
 part.update(bucket_membership_bytes=(folder/'members.bin').stat().st_size,auxiliary_directory_bytes=(folder/'bucket_directory.bin').stat().st_size,bucket_digest_cache_bytes=(folder/'member_digests.bin').stat().st_size,mbt_hashes_bytes=(folder/'tree.bin').stat().st_size,anchor_bytes=(folder/'anchor.json').stat().st_size)
 total=sum(part.values());row={'config':name,'method':'VSQ-V-R2','N':ix.n,'bands':ix.bands,'bits_per_band':ix.width,'nonempty_buckets':ix.tree.L,'membership_occurrences':offset,'record_duplication_multiplier':offset/ix.n,'build_seconds':ix.build_seconds,**part,'index_excluding_vectors_bytes':total,'vector_storage_bytes':(GT/'corpus_vectors.npy').stat().st_size,'total_including_vectors_bytes':total+(GT/'corpus_vectors.npy').stat().st_size}
 save(name+'/storage.json',row)
 # Cache the tree only; vectors and shared headers remain in their original files.
 with (folder/'tree_cache.pkl').open('wb') as f:pickle.dump(ix.tree,f,protocol=5)
 return row

def load_index(name,data):
 a,w=CONFIGS[name];points,sigs,hashes,proj=data;cache=OUT/name/'tree_cache.pkl'
 if not cache.exists():
  ix=Index(points,sigs,hashes,proj,a,w);persist(ix,name);print('INDEX BUILT',name,ix.tree.L,flush=True);return ix
 ix=Index.__new__(Index);ix.points=points;ix.signatures=sigs;ix.hashes=hashes;ix.projection=proj;ix.n,ix.d=points.shape;ix.bands=a;ix.width=w
 with cache.open('rb') as f:ix.tree=pickle.load(f)
 return ix

def reject(wire,q,ix):
 try:verify(wire,q,anchor(ix),ix.projection);return False,'ACCEPTED'
 except Exception as e:return True,repr(e)
def test_index(ix,name):
 tests=[]
 for m in ('image','text'):
  for i in range(20):
   q=QS[m][i];r,p,s,b=ix.response(q);verify(p,q,anchor(ix),ix.projection)
   # Cross-check candidates by scanning band signatures without tree lookups.
   sig=signature(q,ix.projection);step=ix.width//8;mask=np.zeros(ix.n,dtype=bool)
   for band in range(ix.bands):mask|=np.all(ix.signatures[:,band*step:(band+1)*step]==np.frombuffer(sig[band*step:(band+1)*step],dtype='u1'),axis=1)
   assert np.array_equal(np.flatnonzero(mask),r['ids'])
   tests.append({'N':ix.n,'config':name,'modality':m,'query_index':i,'case':'V1','passed':True,'attack_origin':'frozen query','candidate_count':len(r['ids'])})
   if not len(r['result']):
    rid=i+(20 if m=='text' else 0);q=ix.points[rid];r,p,s,b=ix.response(q);verify(p,q,anchor(ix),ix.projection);origin=f'fixed corpus-self fixture rid={rid}; natural scope empty'
   else:origin='frozen query'
   cases={}
   bad=copy.deepcopy(r);bad['vectors'][0,0]+=0.01;cases['V2']=bad
   bad=copy.deepcopy(r);bad['hashes'][0,0]^=1;cases['V3']=bad
   bad=copy.deepcopy(r);bad['signatures'][0,0]^=1;cases['V4']=bad
   li=next(j for j,x in enumerate(r['leaves']) if x[4] is not None and len(x[4]))
   bad=copy.deepcopy(r);x=list(bad['leaves'][li]);x[4][0]=(int(x[4][0])+1)%ix.n;bad['leaves'][li]=x;cases['V5']=bad
   bad=copy.deepcopy(r);x=list(bad['leaves'][li]);x[4]=x[4][1:];x[2]-=1;bad['leaves'][li]=x;cases['V6']=bad
   bad=copy.deepcopy(r);bad['mapping']=bad['mapping'][1:];cases['V7']=bad
   bad=copy.deepcopy(r);x=list(bad['leaves'][li]);x[1]=(x[1][0]+1,x[1][1]);bad['leaves'][li]=x;cases['V8']=bad
   bad=copy.deepcopy(r);lev,pos,h=bad['proof'][0];bad['proof'][0]=(lev,pos,bytes([h[0]^1])+h[1:]);cases['V9']=bad
   bad=copy.deepcopy(r);bad['result']=[next(j for j in range(ix.n) if j not in r['result'])]+r['result'][1:];cases['V11']=bad
   bad=copy.deepcopy(r);bad['result']=bad['result'][1:];cases['V12']=bad
   for case,bad in cases.items():
    ok,err=reject(encode(bad)[0],q,ix);tests.append({'N':ix.n,'config':name,'modality':m,'query_index':i,'case':case,'passed':ok,'error':err,'attack_origin':origin})
   changed=q.copy();changed[0]+=0.01;ok,err=reject(p,changed,ix);tests.append({'N':ix.n,'config':name,'modality':m,'query_index':i,'case':'V10','passed':ok,'error':err,'attack_origin':origin})
   if i==0:assert p==ix.response(q)[1]
 assert all(t['passed'] for t in tests)
 print('CORRECTNESS',name,ix.n,len(tests),'PASS',flush=True)
 return tests

ACTIVE=None;ACTIVE_NAME=None;ACTIVE_RADIUS=0

def one(task):
 m,i,split,timing_mode=task;ix=ACTIVE;q=QS[m][i];truth=set(map(int,GTS[m][i]));r,p,s,b=ix.response(q,radius=ACTIVE_RADIUS)
 vt=verify(p,q,anchor(ix),ix.projection,radius=ACTIVE_RADIUS);recall=len(set(r['result'])&truth)/20;coverage=len(set(map(int,r['ids']))&truth)/20
 assert recall==coverage
 row={'method':'VSQ-V-R2' if ACTIVE_RADIUS==0 else 'VSQ-V-RANGE','config':ACTIVE_NAME,'split':split,'modality':m,'query_index':i,'query_id':QID[m,i],'timing_mode':timing_mode,'valid':True,'recall':recall,'candidate_gt_coverage':coverage,**s,**b,'verification_seconds':vt['total_verification_seconds']}
 row['end_to_end_online_seconds']=s['search_seconds']+s['proof_generation_seconds']+s['serialization_seconds']+vt['total_verification_seconds']
 row['result_ids_sha256']=H(np.asarray(r['result'],dtype='<u4').tobytes()).hex();row['wire_sha256']=H(p).hex()
 vo={'method':row['method'],'config':ACTIVE_NAME,'split':split,'modality':m,'query_index':i,**b}
 ver={'method':row['method'],'config':ACTIVE_NAME,'split':split,'modality':m,'query_index':i,'timing_mode':timing_mode,**vt}
 return row,vo,ver

def execute(ix,name,split,radius=0):
 global ACTIVE,ACTIVE_NAME,ACTIVE_RADIUS
 ACTIVE=ix;ACTIVE_NAME=name;ACTIVE_RADIUS=radius;rows=[];vos=[];vts=[]
 if split=='screening':serial=[(m,i,split,'single_worker') for m in ('image','text') for i in range(100)];parallel=[]
 else:
  serial=[(m,i,split,'single_worker') for m in ('image','text') for i in range(100,150)]
  parallel=[(m,i,split,'throughput_16_workers') for m in ('image','text') for i in range(150,1000)]
 # Warmup is outside measured sample.
 one(serial[0]);gc.collect()
 log=OUT/f'{name}_{split}.jsonl'
 if log.exists():raise RuntimeError('refuse overwriting completed/partial run '+str(log))
 def append(data):
  row,vo,vt=data;rows.append(row);vos.append(vo);vts.append(vt)
  with log.open('a') as f:f.write(json.dumps({'raw':row,'vo':vo,'verification':vt})+'\n')
 for j,t in enumerate(serial):
  append(one(t))
  if j%20==0:print('QUERY',name,split,'single',j,flush=True)
 if parallel:
  started=time.perf_counter()
  with mp.get_context('fork').Pool(min(16,len(os.sched_getaffinity(0)))) as pool:
   for j,result in enumerate(pool.imap_unordered(one,parallel,chunksize=1)):
    append(result)
    if j%100==0:print('QUERY',name,split,'parallel',j,flush=True)
  save(f'{name}_throughput.json',{'queries':len(parallel),'workers':min(16,len(os.sched_getaffinity(0))),'wall_seconds':time.perf_counter()-started,'not_single_query_latency':True})
 order=lambda r:(r['modality'],r['query_index'])
 rows.sort(key=order);vos.sort(key=order);vts.sort(key=order)
 table(f'{name}_{split}_raw.csv',rows);table(f'{name}_{split}_vo.csv',vos);table(f'{name}_{split}_verification.csv',vts)
 save(f'{name}_{split}_complete.json',{'complete':True,'queries':len(rows),'single_worker_latency_queries':len(serial),'radius':radius})
 return rows

def main():
 assert json.loads((ROOT/'results/vsq_original_v1/correctness.json').read_text())['passed']
 data=base_data();tests=[]
 for n in (1000,5000):
  for name,(a,w) in CONFIGS.items():
   ix=Index(data[0][:n],data[1][:n],data[2][:n],data[3],a,w);tests.extend(test_index(ix,name));del ix;gc.collect()
 table('attack_results.csv',tests);save('correctness.json',{'passed':True,'honest_frozen_queries':320,'total_checks':len(tests),'all_attack_rejected':True,'classes':{c:{'trials':sum(t['case']==c for t in tests),'passed':sum(t['case']==c and t['passed'] for t in tests)} for c in ['V'+str(i) for i in range(1,13)]},'empty_scope_policy':'return min(K,scope); nonempty fixed corpus-self fixtures explicitly labeled for attacks'})
 allscreen=[]
 for name in CONFIGS:
  ix=load_index(name,data);allscreen.extend(execute(ix,name,'screening'));del ix;gc.collect()
 table('screening_raw.csv',allscreen)
 stats=[]
 for name in CONFIGS:
  rr=[r for r in allscreen if r['config']==name];stats.append({'config':name,'macro_recall':float(np.mean([r['recall'] for r in rr])),'mean_wire':float(np.mean([r['result_plus_proof_bytes'] for r in rr])),'mean_verify':float(np.mean([r['verification_seconds'] for r in rr]))})
 choice=sorted(stats,key=lambda x:(-x['macro_recall'],x['mean_wire'],x['mean_verify']))[0]
 save('frozen_working_points.json',{'configurations':CONFIGS,'selection_rule':'screening macro recall descending, wire ascending, verification ascending','screening_statistics':stats,'recommended':choice['config'],'holdout_indices':[100,999],'latency_indices':[100,149],'no_holdout_tuning':True})
 low=all(np.mean([r['recall'] for r in allscreen if r['config']==name and r['modality']==m])<.9 for name in CONFIGS for m in ('image','text'))
 save('range_decision.json',{'trigger':'all four configurations have both image/text Recall<.90','triggered':bool(low)})
 if low:
  for name in ('V3','V4'):
   ix=load_index(name,data);execute(ix,name+'_RANGE','screening',radius=1);del ix;gc.collect()
 # Evaluate holdout queries only after freezing the screening selection.
 allhold=[]
 for name in CONFIGS:
  ix=load_index(name,data);allhold.extend(execute(ix,name,'holdout'));del ix;gc.collect()
 table('holdout_raw.csv',allhold);save('complete.json',{'complete':True,'screening_rows':len(allscreen),'holdout_rows':len(allhold),'recommended_from_screening':choice['config']})
if __name__=='__main__':main()

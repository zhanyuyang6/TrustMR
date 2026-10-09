"""TrustMR experiments: compact-protocol evaluation on the VSQ holdout query set."""
import os,sys,time,json,csv,pickle,multiprocessing as mp,gc
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent;ROOT=B.parents[1];OUT=ROOT/'results/vsq_vector_v1';GT=ROOT/'results/evaluation_manifest_v1';OLD=ROOT/'results/vgann_final_screen_v1'
sys.path.insert(0,str(ROOT))
from src import protocol_v6 as ref
from src import protocol_compact as c
from src import compact_wire
STORE=None
QS={m:np.load(GT/f'{m}_queries.npy',mmap_mode='r') for m in ('image','text')};GTS={m:np.load(GT/f'{m}_gt_ids.npy',mmap_mode='r') for m in ('image','text')}
META=list(csv.DictReader((GT/'query_ids.csv').open()));QID={(r['modality'],int(r['query_index'])):r['query_id'] for r in META}
def table(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def query(m,i):return ref.Query(QS[m][i],m,1,0,118286,tuple(s for s,_ in STORE.env.registry),20,1000,8,('image',))
def response(q):
 t=time.perf_counter()
 if not ref.valid_query(q,STORE.env):raise ValueError('invalid query')
 ps=ref.probes(q,STORE.env);buckets=tuple(STORE.buckets[i] for i in ps);rows=ref.filtered((r for b in buckets for r in b.records),q)
 coarse=c.select_coarse(rows,q,STORE.env);openings=tuple((i,STORE.vectors[i]) for i in coarse);result=c.top(openings,q);search=time.perf_counter()-t
 t=time.perf_counter();vo=c.MainResponse(q.commitment(),STORE.env.commitment(),ps,buckets,coarse,result,openings);proof=time.perf_counter()-t
 return vo,search,proof,len(rows)
def one(task):
 m,i,timing=task;q=query(m,i);vo,search,proof,scope=response(q)
 t=time.perf_counter();wire=compact_wire.encode(vo);ser=time.perf_counter()-t
 assert len(wire)==c.wire_size(vo)
 t=time.perf_counter();received=compact_wire.decode(wire);decode=time.perf_counter()-t;v=c.verify(q,received,STORE.env,STORE.ledger);verify=time.perf_counter()-t
 assert v.valid
 resultbytes=len(ref.pack(vo.result))-len(ref.pack(()))
 row={'method':'TrustMR','config':'B2-m8-K1000','split':'holdout','modality':m,'query_index':i,'query_id':QID[m,i],'timing_mode':timing,'valid':True,'recall':len(set(vo.result)&set(map(int,GTS[m][i])))/20,'search_seconds':search,'proof_generation_seconds':proof,'serialization_seconds':ser,'verification_seconds':verify,'end_to_end_online_seconds':search+proof+ser+verify,'unique_candidates':scope,'candidate_fraction':scope/118287,'raw_bucket_memberships':scope,'duplicate_factor':1,'mapped_buckets':len(vo.probe_ids),'cosine_records':len(vo.openings),'result_count':len(vo.result),'proof_only_bytes':len(wire)-resultbytes,'result_plus_proof_bytes':len(wire),'result_ids_sha256':ref.digest('results',ref.pack(vo.result)).hex()}
 vr={'method':'TrustMR','split':'holdout','modality':m,'query_index':i,'timing_mode':timing,'deserialize_seconds':decode,'total_verification_seconds':verify,**{'native_step_'+str(s['step'])+'_seconds':s['ms']/1000 for s in v.steps}}
 return row,vr

def main():
 global STORE
 assert json.loads((OUT/'complete.json').read_text())['complete']
 with (OLD/'trustmr_store.pkl').open('rb') as f:STORE=pickle.load(f)
 assert STORE.env.prefix==8 and STORE.env.projection.shape==(256,1024) and len(STORE.vectors)==118287
 assert np.array_equal(STORE.env.projection,np.load(OUT/'shared/projection.npy'))
 corpus=np.load(GT/'corpus_vectors.npy',mmap_mode='r')
 for i in (0,1000,118286):assert np.array_equal(STORE.vectors[i].view('u4'),corpus[i].view('u4'))
 screening=[]
 for raw in csv.DictReader((OLD/'trustmr_raw.csv').open()):
  row={'method':'TrustMR','config':'B2-m8-K1000','split':'screening','modality':raw['modality'],'query_index':int(raw['query_index']),'query_id':raw['query_id'],'timing_mode':'single_worker_historical','valid':True,'recall':float(raw['recall']),'search_seconds':float(raw['search_seconds']),'proof_generation_seconds':float(raw['proof_generation_seconds']),'serialization_seconds':float(raw['serialize_seconds']),'verification_seconds':float(raw['total_verification_seconds']),'end_to_end_online_seconds':float(raw['end_to_end_with_serialization_seconds']),'unique_candidates':118287,'candidate_fraction':1,'raw_bucket_memberships':118287,'duplicate_factor':1,'mapped_buckets':256,'cosine_records':1000,'result_count':20,'proof_only_bytes':int(raw['proof_only_bytes']),'result_plus_proof_bytes':int(raw['result_plus_proof_bytes'])};screening.append(row)
 table('trustmr_screening_raw.csv',screening);save('trustmr_screening_provenance.json',{'source':str(OLD/'trustmr_raw.csv'),'queries':200,'freshly_remeasured':False,'same_fixed_configuration':True,'single_worker':True,'timing_boundary':'search+proof assembly+serialization+byte-to-verdict verification'})
 rows=[];vr=[];log=OUT/'trustmr_holdout.jsonl'
 if log.exists():raise RuntimeError('refuse overwriting holdout')
 def append(result):
  r,v=result;rows.append(r);vr.append(v)
  with log.open('a') as f:f.write(json.dumps({'raw':r,'verification':v})+'\n')
 for m in ('image','text'):
  q=query(m,100);vo,_,_,_=response(q);assert compact_wire.encode(vo)==compact_wire.encode(c.retrieve(q,STORE));one((m,100,'warmup'));gc.collect()
  for i in range(100,150):
   append(one((m,i,'single_worker')))
   if i%10==0:print('TRUSTMR SINGLE',m,i,flush=True)
 tasks=[(m,i,'throughput_16_workers') for m in ('image','text') for i in range(150,1000)];t=time.perf_counter();workers=min(16,len(os.sched_getaffinity(0)))
 with mp.get_context('fork').Pool(workers) as pool:
  for j,r in enumerate(pool.imap_unordered(one,tasks,chunksize=1)):
   append(r)
   if j%100==0:print('TRUSTMR PARALLEL',j,flush=True)
 save('trustmr_throughput.json',{'queries':len(tasks),'workers':workers,'wall_seconds':time.perf_counter()-t,'not_single_query_latency':True})
 rows.sort(key=lambda r:(r['modality'],r['query_index']));vr.sort(key=lambda r:(r['modality'],r['query_index']));table('trustmr_holdout_raw.csv',rows);table('trustmr_verification_breakdown.csv',vr)
 save('trustmr_complete.json',{'complete':True,'honest_passed':len(rows),'single_worker_latency_queries':100,'historical_screening_queries':200,'index_source':str(OLD/'index_trustmr.json'),'historical_attack_source':str(OLD/'trustmr_attacks.json'),'new_attacks_performed':False})
if __name__=='__main__':main()

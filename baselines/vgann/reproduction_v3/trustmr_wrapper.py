"""TrustMR experiments: compact-protocol measurements on the V-GANN common query set."""
import sys,time,json,csv,struct,hashlib,pickle
from pathlib import Path
from dataclasses import replace
import numpy as np
from run import B,ROOT,OUT,GT,queries,table,save
sys.path.insert(0,str(ROOT))
from src import protocol_v6 as ref
from src import protocol_compact as c
from src import compact_wire

def build():
    cache=OUT/'trustmr_store.pkl'
    if cache.exists():return pickle.loads(cache.read_bytes())
    t=time.perf_counter();env=c.setup(registry=tuple((f'S{i:02}',f'L{i:02}') for i in range(16)))
    vectors=np.load(GT/'corpus_vectors.npy',mmap_mode='r');sigs=ref.signatures(vectors,env);bids=sigs[:,:env.prefix].astype(np.int64)@(1<<np.arange(env.prefix))
    meta=list(csv.DictReader((GT/'corpus_ids.csv').open()));records=[]
    for rid,(v,bid,entry) in enumerate(zip(vectors,bids,meta)):
        raw=(ROOT/'data/raw/coco/train2017'/f"{int(entry['coco_image_id']):012}.jpg").read_bytes()
        records.append(ref.Record(rid,ref.digest('content',raw),v,'image',rid,f'S{rid%16:02}',1,int(bid)))
    print('TRUSTMR COMMIT',flush=True);store=c.commit(env,records);elapsed=time.perf_counter()-t
    paths=[]
    header=OUT/'trustmr_headers.bin'
    with header.open('wb') as f:
        for part in c.disclosure_parts(tuple(store.buckets.values()),()):f.write(part)
    paths.append(header)
    projection=OUT/'trustmr_projection.npy';np.save(projection,env.projection);paths.append(projection)
    anchor=OUT/'trustmr_anchor.json';anchor.write_bytes(ref.pack({'epoch':env.epoch,'environment_hash':env.commitment().hex(),'anchors':[(n,h.hex()) for n,h in store.ledger.anchors],'encoder':env.encoder_id,'registry':env.registry,'modalities':env.modalities,'prefix':env.prefix}));paths.append(anchor)
    # Count the existing frozen-vector file once in persistent storage.
    indexbytes=sum(p.stat().st_size for p in paths)
    save('index_trustmr.json',{'method':'TrustMR','N':len(vectors),'index_only_bytes':indexbytes,'vector_container_bytes':(GT/'corpus_vectors.npy').stat().st_size,'vector_bytes':vectors.nbytes,'total_storage_bytes':indexbytes+(GT/'corpus_vectors.npy').stat().st_size,'build_seconds':elapsed,'storage_files':[str(p) for p in paths],'vector_file':str(GT/'corpus_vectors.npy'),'note':'pickle is an operational cache excluded from logical index/storage; headers include cached leaves; tree interior nodes not retained by B2'})
    cache.write_bytes(pickle.dumps(store,protocol=5));print('TRUSTMR BUILT',elapsed,flush=True)
    return store

def response(q,store):
    # Time candidate search separately from proof assembly.
    t=time.perf_counter()
    if not ref.valid_query(q,store.env):raise ValueError('invalid query')
    ps=ref.probes(q,store.env);buckets=tuple(store.buckets[i] for i in ps)
    rows=ref.filtered((r for b in buckets for r in b.records),q);coarse=c.select_coarse(rows,q,store.env)
    openings=tuple((i,store.vectors[i]) for i in coarse);result=c.top(openings,q);search=time.perf_counter()-t
    t=time.perf_counter();vo=c.MainResponse(q.commitment(),store.env.commitment(),ps,buckets,coarse,result,openings);proof=time.perf_counter()-t
    return vo,search,proof,len(rows)

def verify(q,p,store):
    t=time.perf_counter();tt=t;vo=compact_wire.decode(p);decode=time.perf_counter()-tt
    verdict=c.verify(q,vo,store.env,store.ledger);total=time.perf_counter()-t
    if not verdict.valid:raise ValueError('TrustMR verification failed')
    return {'deserialize_seconds':decode,'total_verification_seconds':total,**{'step_'+str(s['step'])+'_seconds':s['ms']/1000 for s in verdict.steps}}

def attacks(q,vo,store,trial):
    cases={};rid,vec=vo.openings[0];v=vec.copy();v[0]+=0.01
    cases['vector']=replace(vo,openings=((rid,v),)+vo.openings[1:])
    cases['result']=replace(vo,result=(next(i for i in range(len(store.vectors)) if i not in vo.result),)+vo.result[1:])
    cases['missing_opening']=replace(vo,openings=vo.openings[1:])
    cases['routing']=replace(vo,probe_ids=vo.probe_ids[1:])
    bi=next(i for i,b in enumerate(vo.buckets) if b.records);bucket=vo.buckets[bi]
    rr=replace(bucket.records[0],signature=bytes([bucket.records[0].signature[0]^1])+bucket.records[0].signature[1:])
    b=replace(bucket,records=(rr,)+bucket.records[1:]);cases['signature']=replace(vo,buckets=vo.buckets[:bi]+(b,)+vo.buckets[bi+1:])
    b=replace(bucket,records=bucket.records[1:],leaves=bucket.leaves[1:]);cases['missing_record']=replace(vo,buckets=vo.buckets[:bi]+(b,)+vo.buckets[bi+1:])
    cases['query_binding']=replace(vo,query_hash=b'\0'*32)
    cases['environment_binding']=replace(vo,environment_hash=b'\0'*32)
    results=[]
    for kind,bad in cases.items():
        try:verify(q,compact_wire.encode(bad),store);passed=False
        except (ValueError,KeyError,IndexError):passed=True
        results.append({'attack':kind,'trial':trial,'rejected':passed})
    assert all(x['rejected'] for x in results)
    return results

def main():
    assert json.loads((OUT/'vgann_complete.json').read_text())['all_valid']
    assert json.loads((OUT/'independent_reference.json').read_text())['passed']
    assert json.loads((OUT/'sanity.json').read_text())['passed']
    store=build();rows=[];vrows=[];att=[]
    for m,i,qid,qvec,gt in queries(100):
        q=ref.Query(qvec,m,1,0,len(store.vectors)-1,tuple(x for x,_ in store.env.registry),20,1000,8,('image',))
        if i==0:
            warm,_,_,_=response(q,store);canonical=c.retrieve(q,store)
            assert compact_wire.encode(warm)==compact_wire.encode(canonical)
            verify(q,compact_wire.encode(warm),store)
        vo,search,proof,probed=response(q,store)
        t=time.perf_counter();p=compact_wire.encode(vo);serial=time.perf_counter()-t
        assert len(p)==c.wire_size(vo)
        vt=verify(q,p,store)
        # Count result IDs separately; the empty-list bytes belong to framing.
        resultbytes=len(ref.pack(vo.result))-len(ref.pack(()))
        row={'method':'TrustMR','modality':m,'query_index':i,'query_id':qid,'recall':len(set(vo.result)&set(map(int,gt)))/20,'N':len(store.vectors),'K':20,'valid':True,'search_seconds':search,'proof_generation_seconds':proof,'serialize_seconds':serial,**vt,'probed_points':probed,'probed_fraction':probed/len(store.vectors),'ndc':len(vo.openings),'guided_expansions':0,'fallback_expansions':0,'restart_count':0,'proof_only_bytes':len(p)-resultbytes,'result_plus_proof_bytes':len(p),'result_ids':resultbytes,'result_vectors':0,'probed_point_vectors':sum(v.nbytes for _,v in vo.openings)}
        row['end_to_end_seconds']=search+proof+vt['total_verification_seconds'];row['end_to_end_with_serialization_seconds']=row['end_to_end_seconds']+serial
        rows.append(row);vrows.append({'method':'TrustMR','modality':m,'query_index':i,**vt})
        with (OUT/'trustmr_raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if i%5==0:print('TRUSTMR',m,i,row['recall'],'verify',vt['total_verification_seconds'],flush=True)
        if m=='image' and i<20:att.extend(attacks(q,vo,store,i))
    table('trustmr_raw.csv',rows);table('trustmr_verification_breakdown.csv',vrows);save('trustmr_attacks.json',{'honest_passed':200,'attack_rejected':sum(x['rejected'] for x in att),'attack_trials':len(att),'tests':att});save('trustmr_complete.json',{'queries':200,'all_valid':True})
if __name__=='__main__':main()

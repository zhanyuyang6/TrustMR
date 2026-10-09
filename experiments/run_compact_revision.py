"""TrustMR experiments: compact-protocol retrieval, attacks, ablations, and audits."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

from run_v6_revision import load, build, query, save, metric
from src import protocol_v6 as ref
from src import protocol_compact as c


def attacks(vo, q, store):
    yield "honest", vo
    yield "query_binding", replace(vo,query_hash=b"x"*32)
    yield "environment_binding", replace(vo,environment_hash=b"x"*32)
    buckets=vo.buckets[1:]
    rows=ref.filtered((r for b in buckets for r in b.records),q)
    coarse=c.select_coarse(rows,q,store.env)
    openings=tuple((i,store.vectors[i]) for i in coarse)
    yield "routing_joint", replace(vo,buckets=buckets,probe_ids=tuple(b.bid for b in buckets),
                                     coarse_ids=coarse,openings=openings,result=c.top(openings,q))
    yield "bucket_omission", replace(vo,buckets=buckets)
    idx=next(i for i,b in enumerate(vo.buckets) if b.records)
    b=vo.buckets[idx]
    def change_bucket(bucket):
        bs=list(vo.buckets)
        bs[idx]=bucket
        return replace(vo,buckets=tuple(bs))
    yield "member_omission", change_bucket(replace(b,records=b.records[1:],leaves=b.leaves[1:]))
    row=b.records[0]
    for label,field,value in (("timestamp","timestamp",row.timestamp+1000000),
        ("source","source","S15" if row.source!="S15" else "S00"),
        ("modality","modality","text"), ("content_hash","content_hash",b"x"*32),
        ("vector_hash","vector_hash",b"x"*32),
        ("signature","signature",row.signature[:-1]+bytes([row.signature[-1]^128]))):
        bad=replace(row,**{field:value})
        yield label, change_bucket(replace(b,records=(bad,)+b.records[1:]))
    bad=replace(row,timestamp=row.timestamp+1000000)
    yield "joint_tree",change_bucket(replace(b,records=(bad,)+b.records[1:],leaves=(bad.leaf(),)+b.leaves[1:]))
    rid,vec=vo.openings[0]
    yield "vector_opening",replace(vo,openings=((rid,-vec),)+vo.openings[1:])
    yield "missing_opening",replace(vo,openings=vo.openings[1:])
    yield "ranking",replace(vo,result=vo.result[::-1])
    yield "lower_topk",replace(vo,result=c.top(tuple((i,v) for i,v in vo.openings if i!=vo.result[0]),q))
    yield "coarse",replace(vo,coarse_ids=vo.coarse_ids[1:])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--n",type=int,default=118287)
    parser.add_argument("--queries",type=int,default=100)
    parser.add_argument("--verify-queries",type=int,default=20)
    parser.add_argument("--attack-queries",type=int,default=20)
    parser.add_argument("--audit-queries",type=int,default=3)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    save(out/"metadata.json",dict(args={**vars(args),"output":str(out)},
         scope="compact signed-record format; trusted initial signature computation; paired reference results",
         audit="same sampled buckets in reference and compact; disclosed audit vectors remain exhaustive after filtering",
         timing="CPU local snapshot; no consensus/network; wire_size serialization counted separately"))
    vectors,paths,queries=load(args.n)
    env,records,original=build(vectors,paths,out)
    env=c.Environment(**vars(env))
    start=perf_counter()
    store=c.from_validated_store(replace(original,env=env))
    save(out/"compact_ledger.json",dict(environment=env.commitment().hex(),
          anchors=[(n,h.hex()) for n,h in store.ledger.anchors],
          conversion_seconds=perf_counter()-start))
    summary=[]
    for modal,vs in queries.items():
        for m in (4,8):
            for restricted in (False,True):
                label=f"{modal}_m{m}_range{int(restricted)}"
                rows=[]
                for i,v in enumerate(vs[:args.queries]):
                    q=query(v,modal,len(records),m,restricted=restricted)
                    reference=ref.retrieve(q,original)
                    truth=ref.cosine_top(ref.filtered(records,q),q)
                    start=perf_counter()
                    vo=c.retrieve(q,store)
                    elapsed=(perf_counter()-start)*1000
                    assert vo.coarse_ids==reference.coarse_ids and vo.result==reference.result
                    row=dict(i=i,query_ms=elapsed,recall=len(set(vo.result)&set(truth))/len(truth),
                             equivalent=True,opened_vectors=len(vo.openings))
                    if i<args.verify_queries:
                        start=perf_counter()
                        result=c.verify(q,vo,env,store.ledger)
                        row.update(verify_ms=(perf_counter()-start)*1000,valid=result.valid,steps=result.steps)
                        assert result.valid
                        start=perf_counter()
                        row["wire_bytes"]=c.wire_size(vo)
                        row["serialization_count_ms"]=(perf_counter()-start)*1000
                    rows.append(row)
                    with (out/(label+".jsonl")).open("a") as f:
                        f.write(json.dumps(row)+"\n")
                summary.append(dict(label=label,queries=len(rows),metrics={k:metric([r[k] for r in rows if k in r])
                    for k in ("query_ms","recall","opened_vectors","verify_ms","wire_bytes","serialization_count_ms")}))
                save(out/"retrieval_summary.json",summary)
                print("RETRIEVAL",label,"COMPLETE",flush=True)
    counts={}
    for i,v in enumerate(queries["image"][:args.attack_queries]):
        q=query(v,"image",len(records),restricted=True)
        vo=c.retrieve(q,store)
        for name,bad in attacks(vo,q,store):
            for skip in (None,0,1,2,3,4,5):
                verdict=c.verify(q,bad,env,store.ledger,skip_steps=set() if skip is None else {skip})
                key=f"{name}/skip_{skip}"
                counts.setdefault(key,[]).append(dict(valid=verdict.valid,steps=verdict.steps))
                if skip is None:
                    assert verdict.valid == (name=="honest"),name
        save(out/"security_raw.json",counts)
        print("SECURITY",i+1,flush=True)
    audits=[]
    for i,v in enumerate(queries["text"][:args.audit_queries]):
        q=query(v,"text",len(records),restricted=True)
        vo=c.retrieve(q,store)
        start=perf_counter()
        ch=c.challenge(vo,store.ledger,24,20)
        challenge_ms=(perf_counter()-start)*1000
        start=perf_counter()
        audit=c.answer_audit(q,ch,store)
        assembly_ms=(perf_counter()-start)*1000
        start=perf_counter()
        result=c.verify(q,vo,env,store.ledger,ch,audit)
        verify_ms=(perf_counter()-start)*1000
        assert result.valid
        old=ref.retrieve(q,original)
        oldch=ref.challenge(old,original.ledger,24,20,seeds=ch.seeds)
        oldaudit=ref.answer_audit(oldch,original)
        expected=ref.verify(q,old,original.env,original.ledger,oldch,oldaudit)
        assert expected.valid and result.augmented==expected.augmented
        assert result.material_omission==expected.material_omission
        assert not c.verify(q,vo,env,store.ledger,ch,replace(audit,openings=audit.openings[:-1])).valid
        truth=set(ref.cosine_top(ref.filtered(records,q),q))
        audits.append(dict(i=i,valid=True,equivalent=True,material_omission=result.material_omission,
                      challenge_ms=challenge_ms,assembly_ms=assembly_ms,verify_ms=verify_ms,
                      wire_bytes=c.audit_wire_size(audit),opened_vectors=len(audit.openings),
                      seeds=[s.hex() for s in ch.seeds],rounds=ch.rounds,
                      initial_recall=len(truth&set(vo.result))/len(truth),
                      augmented_recall=len(truth&set(result.augmented))/len(truth)))
        save(out/"audit.json",audits)
    save(out/"COMPLETE.json",dict(complete=True))
    print("COMPLETE",out,flush=True)


if __name__=="__main__":
    main()

"""TrustMR experiments: V-GANN correctness gates and common-query measurements."""
import csv,json,time,hashlib,copy,subprocess,resource,os,sys,importlib,shutil,re
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives import serialization
B=Path(__file__).resolve().parent;ROOT=B.parents[2];OUT=ROOT/'results/vgann_final_screen_v1';GT=ROOT/'results/evaluation_manifest_v1';OLD=ROOT/'results/vgann_screen_v2'
META=list(csv.DictReader((GT/'query_ids.csv').open()))
QID={(r['modality'],int(r['query_index'])):r['query_id'] for r in META}
def save(name,v): (OUT/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def table(name,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def queries(count=20):
    for modal in ('image','text'):
        qs=np.load(GT/f'{modal}_queries.npy',mmap_mode='r');gt=np.load(GT/f'{modal}_gt_ids.npy',mmap_mode='r')
        for i in range(count):yield modal,i,QID[modal,i],qs[i],gt[i]
def read_graph(path,n):
    arr=np.fromfile(path,dtype='<i4');pos=0;g=[]
    for _ in range(n):
        length=int(arr[pos]);pos+=1;row=arr[pos:pos+length].copy();pos+=length
        if len(row)!=length or np.any(row<0) or np.any(row>=n):raise ValueError('invalid graph')
        g.append(row)
    assert pos==len(arr)
    return g
CACHE={}
def build(n,core):
    if n in CACHE:
        ix=CACHE[n];ix.__class__=core.Index;return ix
    gp=OUT/f'graph_{n}.ivecs';fp=OLD/f'corpus_{n}.fvecs'
    if not gp.exists():
        if (OLD/gp.name).exists():
            shutil.copyfile(OLD/gp.name,gp);shutil.copyfile(OLD/f'hcnng_{n}_resource.txt',OUT/f'hcnng_{n}_resource.txt')
        else:
            print('BUILD HCNNG',n,flush=True)
            with (OUT/f'hcnng_{n}.log').open('w') as f:
                subprocess.run(['/usr/bin/time','-v','-o',str(OUT/f'hcnng_{n}_resource.txt'),str(B.parent/'reproduction_v2/native/hcnng'),str(fp),'1000','20',str(gp)],stdout=f,stderr=subprocess.STDOUT,check=True)
    graph=read_graph(gp,n);points=np.load(GT/'corpus_vectors.npy',mmap_mode='r')[:n]
    index=core.Index(points,graph,OUT);CACHE[n]=index
    tp=OUT/f'tuples_{n}.bin'
    with tp.open('wb') as f:
        for tup in index.tuplebytes:f.write(core.U32.pack(len(tup)));f.write(tup)
    hp=OUT/f'mht_{n}.bin';hp.write_bytes(b''.join(h for layer in index.levels for h in layer))
    pub=index.public.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
    (OUT/f'anchor_{n}.bin').write_bytes(pub+index.signature)
    save(f'anchor_{n}.json',{'root':index.root.hex(),'signature':index.signature.hex(),'public_key':pub.hex(),'graph_sha256':hashlib.sha256(gp.read_bytes()).hexdigest()})
    rt=(OUT/f'hcnng_{n}_resource.txt').read_text();val=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.*)',rt).group(1)
    elapsed=sum(float(x)*60**j for j,x in enumerate(reversed(val.split(':'))));degrees=[len(g) for g in graph]
    sizes={'N':n,'method':'V-GANN','vector_bytes':n*4096,'vector_container_bytes':fp.stat().st_size,'graph_bytes':gp.stat().st_size,'tuple_bytes':tp.stat().st_size,'mht_bytes':hp.stat().st_size,'anchor_bytes':96,'hcnng_build_seconds':elapsed,'tuple_build_seconds':index.tuple_seconds,'mht_build_seconds':index.mht_seconds,'average_degree':float(np.mean(degrees)),'min_degree':min(degrees),'max_degree':max(degrees),'hcnng_peak_RSS_bytes':int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',rt).group(1))*1024}
    sizes['index_only_bytes']=gp.stat().st_size+tp.stat().st_size+hp.stat().st_size+96;sizes['total_storage_bytes']=sizes['index_only_bytes']+fp.stat().st_size
    save(f'index_{n}.json',sizes);print('BUILT',n,'degree',sizes['average_degree'],flush=True)
    return index

def gate(ix,core):
    rows=[]
    for m,i,qid,q,gt in queries():
        r,p,s,b=ix.response(q,20,qid)
        try:core.verify(p,q,20,qid,ix.n,ix.d,ix.public,ix.root);ok=True;error=''
        except Exception as e:ok=False;error=repr(e)
        rows.append({'variant':core.VARIANT,'N':ix.n,'modality':m,'query_index':i,'query_id':qid,'result_count':len(r['result']),'pass':ok,'error':error,'probed':s['probed_points'],'fallback':s['fallback_expansions'],'restart':s['restart_count']})
    save(f'gate_{core.VARIANT}_{ix.n}.json',rows)
    passed=all(r['pass'] and r['result_count']==20 for r in rows)
    print('GATE',core.VARIANT,ix.n,sum(r['pass'] for r in rows),'/40',flush=True)
    return passed,rows

def attacks(ix,core):
    tests=[];eligible=[]
    for m,i,qid,q,gt in queries():
        r,p,s,b=ix.response(q,20,qid);core.verify(p,q,20,qid,ix.n,ix.d,ix.public,ix.root)
        tests.append({'class':'C1','query_id':qid,'pass':True})
        if s['repair_points']:eligible.append((qid,q.copy(),r,s))
        full=next(j for j,e in enumerate(r['entries']) if not isinstance(e[2],bytes) and e[1] in r['result'])
        collapsed=next(j for j,e in enumerate(r['entries']) if isinstance(e[2],bytes))
        cases={}
        bad=copy.deepcopy(r);v=bytearray(bad['entries'][full][2][0]);v[0]^=1;bad['entries'][full][2][0]=bytes(v);cases['C2']=bad
        bad=copy.deepcopy(r);tu=core.unpack_tuple(bad['entries'][full][2][1]);tu[0][1].append((r['start']+1)%ix.n);bad['entries'][full][2][1]=core.pack_tuple(tu);cases['C3']=bad
        bad=copy.deepcopy(r);bad['result'][0]=next(j for j in range(ix.n) if j not in r['result']);cases['C4']=bad
        bad=copy.deepcopy(r);lev,rid,data=bad['entries'][full];bad['entries'][full]=(lev,rid,core.leaf(rid,*data));cases['C5']=bad
        bad=copy.deepcopy(r);bad['start']=(r['start']+1)%ix.n;cases['C6']=bad
        bad=copy.deepcopy(r);lev,rid,data=bad['entries'][collapsed];bad['entries'][collapsed]=(lev,rid,bytes([data[0]^1])+data[1:]);cases['C7']=bad
        bad=copy.deepcopy(r)
        if len(r['restarts'])>=2:bad['restarts']=r['restarts'][::-1]
        elif r['restarts']:bad['restarts']=[(r['restarts'][0]+1)%ix.n]
        else:bad['restarts']=[r['start'],(r['start']+1)%ix.n]
        cases['C9']=bad
        for kind,bad in cases.items():
            try:core.verify(core.encode(bad)[0],q,20,qid,ix.n,ix.d,ix.public,ix.root);ok=False;err='ACCEPTED'
            except Exception as e:ok=True;err=repr(e)
            tests.append({'class':kind,'query_id':qid,'pass':ok,'error':err,'mutation':('reorder restarts; if <2 alter/inject forbidden sequence' if kind=='C9' else kind)})
        if i<2:assert p==ix.response(q,20,qid)[1]
    assert eligible,'C8 requires real repair-opened points'
    for trial in range(20):
        qid,q,r,s=eligible[trial%len(eligible)];bad=copy.deepcopy(r);rid=s['repair_points'][trial%len(s['repair_points'])]
        j=next(j for j,e in enumerate(bad['entries']) if e[1]==rid and not isinstance(e[2],bytes));lev,rid,data=bad['entries'][j];bad['entries'][j]=(lev,rid,core.leaf(rid,*data))
        try:core.verify(core.encode(bad)[0],q,20,qid,ix.n,ix.d,ix.public,ix.root);ok=False;err='ACCEPTED'
        except Exception as e:ok=True;err=repr(e)
        tests.append({'class':'C8','query_id':qid,'trial':trial,'pass':ok,'error':err,'removed_rid':rid})
    report={'passed':all(t['pass'] for t in tests),'classes':{kind:{'trials':sum(t['class']==kind for t in tests),'passed':sum(t['class']==kind and t['pass'] for t in tests)} for kind in ['C'+str(i) for i in range(1,10)]},'C8_eligible_distinct_queries':len(eligible),'C9_note':'Actual reorder if >=2 restarts; otherwise altered/injected non-replayable restart transcript. Does not claim testing two real restarts where none occur.','tests':tests}
    save('attacks.json',report);assert report['passed'];print('ATTACKS PASS',report['classes'],flush=True)
    return report

def screen(ix,core):
    r1=importlib.import_module('core_r1');rows=[];vos=[];vts=[];over=[];starts=[];restarts=[]
    for m,i,qid,q,gt in queries(100):
        r,p,s,b=ix.response(q,20,qid);v=core.verify(p,q,20,qid,ix.n,ix.d,ix.public,ix.root)
        starts.append({'modality':m,'query_index':i,'query_id':qid,'N':ix.n,'initial_rid':r['start']})
        restarts.extend({'modality':m,'query_index':i,'query_id':qid,'counter':j,'rid':rid} for j,rid in enumerate(s['restart_points']))
        row={'method':'V-GANN-'+core.VARIANT,'modality':m,'query_index':i,'query_id':qid,'recall':len(set(r['result'])&set(map(int,gt)))/20,'N':ix.n,'K':20,'valid':True,**{k:x for k,x in s.items() if k not in ('restart_points','repair_points')},**b,**v}
        row['proof_generation_seconds']=s['raw_vo_seconds']+s['collapse_seconds'];row['probed_fraction']=s['probed_points']/ix.n;row['speedup']=ix.n/s['ndc']
        row['end_to_end_seconds']=s['search_seconds']+row['proof_generation_seconds']+v['total_verification_seconds'];row['end_to_end_with_serialization_seconds']=row['end_to_end_seconds']+s['serialize_seconds']
        rows.append(row);vos.append({'method':row['method'],'modality':m,'query_index':i,**b});vts.append({'method':row['method'],'modality':m,'query_index':i,**v})
        # Measure R1 overhead on the same graph and query, separately from variant selection.
        t=time.perf_counter();rr,vv,ss=r1.search(ix.points,ix.tuples,q,20,r['start'],qid,ix.n);st=time.perf_counter()-t
        ee,ps=ix.proof(vv);base=dict(r,entries=ee,result=rr,variant='R1',restarts=[]);_,bp=r1.encode(base)
        over.append({'modality':m,'query_index':i,'query_id':qid,'baseline':'R1','baseline_result_count':len(rr),'fallback_activated':s['fallback_expansions']>0,'restart_activated':s['restart_count']>0,'repair_needed_for_cardinality':len(rr)<20,'extra_probed_points':s['probed_points']-ss['probed_points'],'extra_vo_bytes':b['result_plus_proof_bytes']-bp['result_plus_proof_bytes'],'extra_search_seconds':s['search_seconds']-st,'extra_proof_seconds':row['proof_generation_seconds']-ps['raw_vo_seconds']-ps['collapse_seconds'],'selected_fallback_expansions':s['fallback_expansions'],'selected_restart_count':s['restart_count']})
        with (OUT/'vgann_raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if i%10==0:print('SCREEN',m,i,'recall',row['recall'],'V',s['probed_points'],flush=True)
        if i==0:
            assert ix.response(q,20,qid)[1]==p
            (OUT/f'example_{m}.bin').write_bytes(p)
    table('vgann_raw.csv',rows);table('vo_breakdown.csv',vos);table('verification_breakdown.csv',vts);table('repair_overhead.csv',over);table('initial_points.csv',starts);table('restart_points.csv',restarts,['modality','query_index','query_id','counter','rid'])
    save('vgann_complete.json',{'queries':200,'all_valid':True,'variant':core.VARIANT})

def main():
    history=[];chosen=None
    for variant in ('R1','R2','R3'):
        core=importlib.import_module('core_'+variant.lower());ok=True
        for n in (1000,5000):
            passed,rows=gate(build(n,core),core);history.append({'variant':variant,'N':n,'pass_count':sum(r['pass'] for r in rows),'total':40});ok=ok and passed
        if not ok:continue
        for n in (20000,118287):
            passed,rows=gate(build(n,core),core);history.append({'variant':variant,'N':n,'pass_count':sum(r['pass'] for r in rows),'total':40})
            if not passed:ok=False;break
        if ok:chosen=core;break
    if chosen is None:raise RuntimeError('No V-GANN variant passed all correctness gates')
    selection={'variant':'V-GANN-'+chosen.VARIANT,'label':'V-GANN-R (our repaired reproduction)','reason':'First repair in R1/R2/R3 ladder passing all 40-query gates at all four scales; no recall used.','repair_description':'R1 empty R and guarded max; R2 all original neighbors only at guided dead-end while R<K.' if chosen.VARIANT=='R2' else 'See repair_log.md','honest_pass_rate':1.0,'history':history}
    (B/'selected_variant.json').write_text(json.dumps(selection,indent=2));save('selected_variant.json',selection)
    ix=build(118287,chosen);attack=attacks(ix,chosen)
    save('correctness.json',{'passed':True,'history':history,'attacks':attack['classes'],'selected_variant':selection['variant'],'deterministic_rerun':True})
    screen(ix,chosen)
if __name__=='__main__':main()

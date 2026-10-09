"""TrustMR experiments: VSQ reproduction on set-valued MinHash and Jaccard queries."""
import sys,time,json,csv,struct,copy,difflib
from pathlib import Path
import numpy as np
import msgpack
B=Path(__file__).resolve().parents[1];ROOT=B.parents[1];OUT=ROOT/'results/vsq_original_v1'
sys.path.insert(0,str(B/'vector'))
from core import H,Tree,leaf,recover_root,parent
pack=lambda x:msgpack.packb(x,use_bin_type=True)
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def table(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def words(s):return b' '.join(('w%04d'%int(i)).encode() for i in sorted(s))
class MinHash:
 def __init__(self):
  raw=np.empty((128,5000),dtype='S32');self.ranks=np.empty((5000,128),dtype='<u2')
  for j in range(128):
   raw[j]=[H(b'VSQ-MINHASH'+struct.pack('<H',j)+('w%04d'%i).encode()) for i in range(5000)]
   order=np.argsort(raw[j]);self.ranks[order,j]=np.arange(5000);raw[j]=raw[j,order]
  self.values=raw.view('u1').reshape(128,5000,32)
 def __call__(self,s):
  rank=self.ranks[list(s)].min(axis=0);return self.values[np.arange(128),rank].tobytes()
 def direct(self,s):return b''.join(min(H(b'VSQ-MINHASH'+struct.pack('<H',j)+('w%04d'%i).encode()) for i in s) for j in range(128))
def keys(sig,count,qualified=True):return sorted(set(((j if qualified else 0),int.from_bytes(H(b'BAND'+sig[j*512:(j+1)*512]),'big')%count) for j in range(8)))
def merkle(data):
 layer=[H(b'O'+pack(x)) for x in data];n=len(layer);size=1<<(n-1).bit_length();layer.extend(H(b'P'+struct.pack('<I',i)) for i in range(n,size));levels=[layer]
 while len(layer)>1:layer=[parent(layer[i],layer[i+1]) for i in range(0,len(layer),2)];levels.append(layer)
 return levels
class Original:
 def __init__(self,sets,mh,method):
  self.sets=sets;self.n=len(sets);self.mh=mh;self.method=method;self.bucket_count=1<<(self.n-1).bit_length()
  t=time.perf_counter();self.sigs=[mh(s) for s in sets] if method!='Baseline' else None
  self.records=[(i,i,words(s) if method=='Baseline' else self.sigs[i]) for i,s in enumerate(sets)]
  if method=='VSQ-O-R2':
   groups={}
   for i,sig in enumerate(self.sigs):
    for key in keys(sig,self.bucket_count):groups.setdefault(key,[]).append(i)
   kk=sorted(groups);commits=[H(b'O'+pack(r)) for r in self.records];self.tree=Tree(kk,[np.asarray(groups[k],dtype='<u4') for k in kk],commits);self.root=self.tree.root
  else:self.levels=merkle(self.records);self.root=self.levels[-1][0]
  self.build_seconds=time.perf_counter()-t
 def compute(self,ids,q,qs):
  if self.method=='Baseline':return [(len(self.sets[i]&q)/len(self.sets[i]|q)) for i in ids]
  return [np.mean(np.frombuffer(self.sigs[i],dtype='S32')==np.frombuffer(qs,dtype='S32')) for i in ids]
 def response(self,q,mode='topk',k=None,mu=.9):
  k=k or self.n//10;t=time.perf_counter();qs=self.mh(q) if self.method!='Baseline' else None
  if self.method=='VSQ-O-R2':
   mapping,needed,full=self.tree.locate(keys(qs,self.bucket_count));ids=sorted({int(i) for p in full for i in self.tree.members[p]})
  else:ids=list(range(self.n))
  sim=self.compute(ids,q,qs);ordered=sorted(zip(sim,ids),key=lambda x:(-x[0],x[1]));result=[i for s,i in ordered[:k]] if mode=='topk' else [i for s,i in ordered if s>=mu];search=time.perf_counter()-t
  t=time.perf_counter();r={'method':self.method,'query_hash':H(words(q)),'mode':mode,'k':k,'mu':mu,'result':result,'records':[self.records[i] for i in ids]}
  if self.method=='VSQ-O-R2':r.update(mapping=mapping,leaves=[(i,key,c,d,None if ids is None else ids.tolist()) for i,key,c,d,ids in self.tree.disclosure(needed,full)],proof=self.tree.proof(needed))
  else:r['paths']=[(rid,[self.levels[l][(rid>>l)^1] for l in range(len(self.levels)-1)]) for rid in result]
  gen=time.perf_counter()-t;t=time.perf_counter();wire=pack(r);ser=time.perf_counter()-t
  return r,wire,{'search_seconds':search,'proof_generation_seconds':gen,'serialization_seconds':ser,'candidate_count':len(ids),'vo_bytes':len(wire)}
 def verify(self,wire,q,mode='topk',k=None,mu=.9):
  total=time.perf_counter();r=msgpack.unpackb(wire,raw=False);k=k or self.n//10
  if (r['method'],r['query_hash'],r['mode'],r['k'],r['mu'])!=(self.method,H(words(q)),mode,k,mu):raise ValueError('query binding')
  records={x[0]:x for x in r['records']}
  if len(records)!=len(r['records']) or any(i<0 or i>=self.n for i in records):raise ValueError('record identity')
  qs=self.mh(q) if self.method!='Baseline' else None
  if self.method=='VSQ-O-R2':
   expected=keys(qs,self.bucket_count)
   if [tuple(x[0]) for x in r['mapping']]!=expected:raise ValueError('mapped scope')
   leaves={x[0]:x for x in r['leaves']};required=set();full=set()
   for key,lo,hi in r['mapping']:
    key=tuple(key)
    if lo==hi:
     if lo not in leaves or tuple(leaves[lo][1])!=key:raise ValueError('mapped leaf')
     full.add(lo);required.add(lo)
    else:
     if hi!=lo+1 or lo< -1 or hi>self.tree.L:raise ValueError('absence')
     if lo>=0:
      if lo not in leaves or not tuple(leaves[lo][1])<key:raise ValueError('lower boundary')
      required.add(lo)
     if hi<self.tree.L:
      if hi not in leaves or not key<tuple(leaves[hi][1]):raise ValueError('upper boundary')
      required.add(hi)
   if set(leaves)!=required or {i for i,x in leaves.items() if x[4] is not None}!=full:raise ValueError('leaf scope')
   union=set();lh={}
   for i,key,count,digest,ids in r['leaves']:
    if ids is not None:
     if ids!=sorted(set(ids)) or len(ids)!=count:raise ValueError('members')
     union.update(ids)
     computed=H(b'M'+b''.join(H(b'O'+pack(records[rid])) for rid in ids))
     if computed!=digest:raise ValueError('member commitment')
    lh[i]=leaf(tuple(key),count,digest)
   if union!=set(records):raise ValueError('record dictionary')
   if recover_root(lh,r['proof'],self.tree.L)!=self.root:raise ValueError('root')
  else:
   if set(records)!=set(range(self.n)):raise ValueError('missing mismatch proof')
   if merkle([records[i] for i in range(self.n)])[-1][0]!=self.root:raise ValueError('root')
   if [p[0] for p in r['paths']]!=r['result']:raise ValueError('result paths')
   for rid,path in r['paths']:
    value=H(b'O'+pack(records[rid]))
    for l,sib in enumerate(path):value=parent(sib,value) if (rid>>l)&1 else parent(value,sib)
    if value!=self.root:raise ValueError('result proof')
  sim=[]
  for rid,ts,data in r['records']:
   if self.method=='Baseline':s=frozenset(int(w[1:]) for w in data.split());score=len(s&q)/len(s|q)
   else:
    if len(data)!=4096:raise ValueError('signature width')
    score=np.mean(np.frombuffer(data,dtype='S32')==np.frombuffer(qs,dtype='S32'))
   sim.append((score,rid))
  sim.sort(key=lambda x:(-x[0],x[1]));result=[i for s,i in sim[:k]] if mode=='topk' else [i for s,i in sim if s>=mu]
  if result!=r['result']:raise ValueError('result mismatch')
  return time.perf_counter()-total

def attacks(ix,queries):
 rows=[]
 for j,q in enumerate(queries):
  r,p,st=ix.response(q);ix.verify(p,q);rows.append({'N':ix.n,'trial':j,'case':'O1','passed':True,'fixture':'fixed stored set'})
  cases={}
  b=copy.deepcopy(r);b['records'][0]=list(b['records'][0]);b['records'][0][1]+=1;cases['O2']=b
  b=copy.deepcopy(r);rec=list(b['records'][0]);rec[2]=bytes([rec[2][0]^1])+rec[2][1:];b['records'][0]=rec;cases['O3']=b
  f=next(i for i,x in enumerate(r['leaves']) if x[4]);sib=0
  b=copy.deepcopy(r);x=list(b['leaves'][f]);x[4]=list(x[4]);x[4][0]=(x[4][0]+1)%ix.n;b['leaves'][f]=x;cases['O4']=b
  b=copy.deepcopy(r);x=list(b['leaves'][f]);x[4]=x[4][1:];x[2]-=1;b['leaves'][f]=x;cases['O5']=b
  b=copy.deepcopy(r);b['mapping']=b['mapping'][1:];cases['O6']=b
  b=copy.deepcopy(r);l,i,h=b['proof'][sib];b['proof'][sib]=(l,i,bytes([h[0]^1])+h[1:]);cases['O7']=b
  b=copy.deepcopy(r);x=list(b['leaves'][f]);x[1]=(x[1][0]+1,x[1][1]);b['leaves'][f]=x;cases['O8']=b
  b=copy.deepcopy(r);b['result']=[next(i for i in range(ix.n) if i not in r['result'])]+r['result'][1:];cases['O9']=b
  for case,bad in cases.items():
   try:ix.verify(pack(bad),q);ok=False;err='accepted'
   except Exception as e:ok=True;err=repr(e)
   rows.append({'N':ix.n,'trial':j,'case':case,'passed':ok,'error':err})
  changed=set(q);changed.remove(min(changed))
  try:ix.verify(p,changed);ok=False
  except Exception:ok=True
  rows.append({'N':ix.n,'trial':j,'case':'O10','passed':ok})
 return rows

def main():
 rng=np.random.default_rng(20260928);sets=[frozenset(map(int,rng.choice(5000,1500,replace=False))) for _ in range(4096)];mh=MinHash()
 np.save(OUT/'public_hash_values.npy',mh.values);np.save(OUT/'public_hash_ranks.npy',mh.ranks)
 assert mh(sets[0])==mh.direct(sets[0]);cache=mh.values.nbytes+mh.ranks.nbytes
 qsets=[]
 for j,s in enumerate(sets[:20]):
  if j%2:q=set(s)-set(sorted(s)[:75]);q.update(i for i in sorted(set(range(5000))-set(s))[:75]);qsets.append(frozenset(q))
  else:qsets.append(s)
 # Check result collection and cross-band key collisions independently of timing.
 rtemp=[];stemp=[];combined=list(zip(stemp,rtemp));rtemp.append(1);stemp.append(1.0)
 save('literal_diagnostics.json',{'R0_early_zip_result':combined,'R1_collect_then_zip':list(zip(stemp,rtemp)),'band_identity_fixture':{'records':{'0':[[0,7]],'1':[[1,7]]},'query_key':[0,7],'literal_merged_scope':[0,1],'qualified_scope':[0]},'required_repairs':['collect then sort','band identity','correct sibling hashes and authenticated missing buckets']})
 alltests=[];rows=[];storage=[]
 for n in (128,512,2048,4096):
  truth=[]
  for q in qsets:
   truth.append([i for _,i in sorted(((len(s&q)/len(s|q),i) for i,s in enumerate(sets[:n])),key=lambda x:(-x[0],x[1]))[:n//10]])
  for method in ('Baseline','Baseline+','VSQ-O-R2'):
   ix=Original(sets[:n],mh,method)
   path=OUT/f'ads_{method.replace("+","plus")}_{n}.bin'
   with path.open('wb') as f:
    if method=='VSQ-O-R2':
     for key,group in zip(ix.tree.keys,ix.tree.members):data=pack([key,[ix.records[int(i)] for i in group]]);f.write(struct.pack('<I',len(data)));f.write(data)
     f.write(b''.join(h for layer in ix.tree.levels for h in layer))
    else:
     for rec in ix.records:data=pack(rec);f.write(struct.pack('<I',len(data)));f.write(data)
     f.write(b''.join(h for layer in ix.levels for h in layer))
   storage.append({'N':n,'method':method,'ads_bytes':path.stat().st_size,'public_minhash_cache_bytes':cache if method!='Baseline' else 0,'total_persistent_bytes':path.stat().st_size+(cache if method!='Baseline' else 0),'build_seconds':ix.build_seconds})
   for mode in ('topk','range'):
    for j,q in enumerate(qsets):
     r,p,s=ix.response(q,mode);v=ix.verify(p,q,mode)
     rows.append({'N':n,'method':method,'mode':mode,'query_index':j,'valid':True,**s,'verification_seconds':v,'result_count':len(r['result']),'global_Jaccard_recall':len(set(r['result'])&set(truth[j]))/(n//10) if mode=='topk' else '', 'ads_bytes':path.stat().st_size,'persistent_with_public_cache_bytes':storage[-1]['total_persistent_bytes']})
   if method=='VSQ-O-R2' and n<=2048:alltests.extend(attacks(ix,sets[:20]))
   print('ORIGINAL',n,method,'done',flush=True)
 table('trend.csv',rows);table('original_trend.csv',rows);table('storage.csv',storage);table('attack_results.csv',alltests)
 assert all(t['passed'] for t in alltests)
 save('correctness.json',{'passed':True,'honest_trend_queries':len(rows),'total_checks':len(alltests),'honest_fixtures':sum(t['case']=='O1' for t in alltests),'attack_checks':sum(t['case']!='O1' for t in alltests),'classes':{c:{'trials':sum(t['case']==c for t in alltests),'passed':sum(t['case']==c and t['passed'] for t in alltests)} for c in ['O'+str(i) for i in range(1,11)]},'direct_SHA256_matches_public_cache':True,'label':'VSQ-O-R2 (our repaired original reproduction)'})
 summary=[]
 for n in (128,512,2048,4096):
  for mode in ('topk','range'):
   for method in ('Baseline','Baseline+','VSQ-O-R2'):
    rr=[r for r in rows if r['N']==n and r['mode']==mode and r['method']==method];summary.append({'N':n,'mode':mode,'method':method,**{k:float(np.mean([r[k] for r in rr])) for k in ('search_seconds','verification_seconds','vo_bytes','ads_bytes','candidate_count')},'global_Jaccard_recall':float(np.mean([r['global_Jaccard_recall'] for r in rr])) if mode=='topk' else ''})
 table('summary.csv',summary)
 save('complete.json',{'complete':True,'queries':len(rows),'GT_scope':'small original synthetic sets only; frozen vector GT untouched'})
if __name__=='__main__':main()

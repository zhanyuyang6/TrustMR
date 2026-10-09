"""TrustMR experiments: VSQ vector adaptation with authenticated band-bucket search."""
import hashlib,struct,time,bisect
import numpy as np
H=lambda b:hashlib.sha256(b).digest()
U32=struct.Struct('<I'); KEY=struct.Struct('<HQ'); MAGIC=b'VSQVR002'
def vh(v):return H(b'V'+np.asarray(v,dtype='<f4').tobytes())
def rh(rid,sig,hashv):return H(b'R'+U32.pack(int(rid))+bytes(sig)+bytes(hashv))
def leaf(key,count,digest):return H(b'L'+KEY.pack(*key)+U32.pack(count)+digest)
def parent(a,b):return H(b'I'+a+b)
def signature(q,projection):return np.packbits(np.einsum('ij,j->i',projection,q,dtype=np.float64,optimize=False)>=0,bitorder='little').tobytes()
def bandkeys(sig,bands,width,radius=0):
    step=width//8;keys=[]
    for band in range(bands):
        val=int.from_bytes(sig[band*step:(band+1)*step],'little')
        vals=[val] if radius==0 else sorted([val]+[val^(1<<i) for i in range(width)])
        keys.extend((band,v) for v in vals)
    return keys

def scores(vectors,q):return np.rint(np.einsum('ij,j->i',vectors,q,dtype=np.float64,optimize=False)*1e8).astype(np.int64)
def ranking(ids,score,k):return [int(ids[i]) for i in np.lexsort((ids,-score))[:k]]

class Tree:
    def __init__(self,keys,members,commits):
        self.keys=keys;self.members=members;self.L=len(keys);self.size=1<<(self.L-1).bit_length()
        self.digests=[H(b'M'+b''.join(commits[int(r)] for r in group)) for group in members]
        level=[leaf(key,len(group),d) for key,group,d in zip(keys,members,self.digests)]
        level.extend(H(b'P'+U32.pack(i)) for i in range(self.L,self.size));self.levels=[level]
        while len(level)>1:level=[parent(level[i],level[i+1]) for i in range(0,len(level),2)];self.levels.append(level)
        self.root=level[0]
    def locate(self,keys):
        mapping=[];needed=set();full=set()
        for key in keys:
            j=bisect.bisect_left(self.keys,key)
            if j<self.L and self.keys[j]==key:lo=hi=j;full.add(j);needed.add(j)
            else:
                lo=j-1;hi=j
                if lo>=0:needed.add(lo)
                if hi<self.L:needed.add(hi)
            mapping.append((key,lo,hi))
        return mapping,needed,full
    def proof(self,needed):
        frontier=[];current=set(needed)
        for level in range(len(self.levels)-1):
            for idx in sorted(current):
                if idx^1 not in current:frontier.append((level,idx^1,self.levels[level][idx^1]))
            current={i//2 for i in current}
        return frontier
    def disclosure(self,needed,full):
        return [(i,self.keys[i],len(self.members[i]),self.digests[i],self.members[i] if i in full else None) for i in sorted(needed)]

def recover_root(leafhash,proof,L):
    provided={(int(l),int(i)):h for l,i,h in proof}
    if len(provided)!=len(proof):raise ValueError('duplicate Merkle sibling')
    current=dict(leafhash);used=set();height=(L-1).bit_length()
    if not current:raise ValueError('empty authentication frontier')
    for level in range(height):
        parents={}
        for p in sorted({i//2 for i in current}):
            children=[]
            for i in (2*p,2*p+1):
                if i in current:children.append(current[i])
                else:
                    key=(level,i)
                    if key not in provided:raise ValueError('missing sibling')
                    children.append(provided[key]);used.add(key)
            parents[p]=parent(*children)
        current=parents
    if set(current)!={0} or used!=set(provided):raise ValueError('invalid proof frontier')
    return current[0]

class Index:
    def __init__(self,points,signatures,hashes,projection,bands,width):
        self.points=points;self.signatures=signatures;self.hashes=hashes;self.projection=projection
        self.n,self.d=points.shape;self.bands=bands;self.width=width
        t=time.perf_counter();commits=[rh(i,signatures[i],hashes[i]) for i in range(self.n)];self.commits=commits
        keys=[];members=[];step=width//8;ids=np.arange(self.n,dtype='<u4')
        for band in range(bands):
            vals=np.ascontiguousarray(signatures[:,band*step:(band+1)*step]).view('<u'+str(step)).reshape(-1)
            order=np.lexsort((ids,vals));sortedvals=vals[order];cuts=np.r_[0,np.flatnonzero(sortedvals[1:]!=sortedvals[:-1])+1,self.n]
            for a,b in zip(cuts[:-1],cuts[1:]):keys.append((band,int(sortedvals[a])));members.append(ids[order[a:b]])
        self.tree=Tree(keys,members,commits);self.build_seconds=time.perf_counter()-t
    def response(self,q,k=20,radius=0):
        total=time.perf_counter();stats={};t=time.perf_counter();qs=signature(q,self.projection);stats['query_signature_seconds']=time.perf_counter()-t
        t=time.perf_counter();keys=bandkeys(qs,self.bands,self.width,radius);stats['band_hashing_seconds']=time.perf_counter()-t
        t=time.perf_counter();mapping,needed,full=self.tree.locate(keys);stats['mbt_lookup_seconds']=time.perf_counter()-t
        t=time.perf_counter();groups=[self.tree.members[i] for i in sorted(full)];raw=sum(len(g) for g in groups);flat=np.concatenate(groups) if groups else np.empty(0,dtype='<u4');stats['bucket_materialization_seconds']=time.perf_counter()-t
        t=time.perf_counter();ids=np.unique(flat).astype('<u4',copy=False);stats['dedup_seconds']=time.perf_counter()-t
        t=time.perf_counter();vectors=self.points[ids];score=scores(vectors,q);stats['cosine_seconds']=time.perf_counter()-t
        t=time.perf_counter();result=ranking(ids,score,k);stats['topk_seconds']=time.perf_counter()-t;stats['search_seconds']=time.perf_counter()-total
        t=time.perf_counter();r={'n':self.n,'bands':self.bands,'width':self.width,'k':k,'L':self.tree.L,'radius':radius,'query_hash':vh(q),'mapping':mapping,'leaves':self.tree.disclosure(needed,full),'proof':self.tree.proof(needed),'ids':ids,'signatures':self.signatures[ids],'hashes':self.hashes[ids],'vectors':vectors,'result':result}
        stats['proof_generation_seconds']=time.perf_counter()-t
        t=time.perf_counter();wire,parts=encode(r);stats['serialization_seconds']=time.perf_counter()-t
        stats.update(mapped_bands=self.bands,mapped_buckets=len(keys),nonempty_mapped_buckets=len(full),absence_buckets=len(keys)-len(full),boundary_leaf_summaries=len(needed-full),raw_bucket_memberships=raw,unique_candidates=len(ids),candidate_fraction=len(ids)/self.n,duplicate_factor=raw/len(ids) if len(ids) else 0,result_count=len(result))
        parts['raw_logical_payload_bytes']=len(wire)+(raw-len(ids))*(4+32+32+self.d*4)
        return r,wire,stats,parts

def encode(r):
    chunks=[];parts={k:0 for k in ('merkle_sibling_hashes','bucket_member_digests','bucket_ids','bucket_member_rids','unique_record_ids','unique_signatures','vector_hashes','opened_vectors','structural_framing','results','metadata')}
    def put(b,name):chunks.append(b);parts[name]+=len(b)
    put(MAGIC+struct.pack('<IHHIIB',r['n'],r['bands'],r['width'],r['k'],r['L'],r['radius'])+r['query_hash'],'metadata')
    put(U32.pack(len(r['mapping'])),'structural_framing')
    for key,lo,hi in r['mapping']:put(KEY.pack(*key),'bucket_ids');put(struct.pack('<ii',lo,hi),'structural_framing')
    put(U32.pack(len(r['leaves'])),'structural_framing')
    for i,key,count,digest,rids in r['leaves']:
        put(struct.pack('<IIB',i,count,rids is not None),'structural_framing');put(KEY.pack(*key),'bucket_ids');put(digest,'bucket_member_digests')
        if rids is not None:put(np.asarray(rids,dtype='<u4').tobytes(),'bucket_member_rids')
    put(U32.pack(len(r['proof'])),'structural_framing')
    for level,i,h in r['proof']:put(struct.pack('<HI',level,i),'structural_framing');put(h,'merkle_sibling_hashes')
    put(U32.pack(len(r['ids'])),'structural_framing');put(np.asarray(r['ids'],dtype='<u4').tobytes(),'unique_record_ids')
    put(np.asarray(r['signatures'],dtype='u1').tobytes(),'unique_signatures');put(np.asarray(r['hashes'],dtype='u1').tobytes(),'vector_hashes');put(np.asarray(r['vectors'],dtype='<f4').tobytes(),'opened_vectors')
    put(U32.pack(len(r['result']))+np.asarray(r['result'],dtype='<u4').tobytes(),'results')
    wire=b''.join(chunks);assert sum(parts.values())==len(wire)
    parts['proof_only_bytes']=len(wire)-parts['results'];parts['result_plus_proof_bytes']=len(wire)
    return wire,parts

def decode(wire,d=1024):
    pos=0
    def take(n):
        nonlocal pos
        if n<0 or pos+n>len(wire):raise ValueError('truncated wire')
        x=memoryview(wire)[pos:pos+n];pos+=n;return x
    def u32():return U32.unpack(take(4))[0]
    if take(8)!=MAGIC:raise ValueError('wire magic')
    n,bands,width,k,L,radius=struct.unpack('<IHHIIB',take(17));qh=bytes(take(32))
    if not 0<n<=10**7 or not 0<L<=32*n or bands*width!=256 or width not in (8,16,32,64) or not 0<k<=n or radius not in (0,1):raise ValueError('wire parameters')
    nm=u32()
    if nm!=bands*(1 if radius==0 else width+1):raise ValueError('mapped count')
    mapping=[(KEY.unpack(take(10)),*struct.unpack('<ii',take(8))) for _ in range(nm)]
    nl=u32()
    if nl>2*nm:raise ValueError('leaf count')
    leaves=[]
    for _ in range(nl):
        idx,count,full=struct.unpack('<IIB',take(9));key=KEY.unpack(take(10));digest=bytes(take(32))
        if idx>=L or count>n or full not in (0,1):raise ValueError('leaf parameters')
        rids=np.frombuffer(take(4*count),dtype='<u4') if full else None;leaves.append((idx,key,count,digest,rids))
    npf=u32()
    if npf>2*nm*((L-1).bit_length()+1):raise ValueError('proof count')
    proof=[(*struct.unpack('<HI',take(6)),bytes(take(32))) for _ in range(npf)]
    U=u32()
    if U>n:raise ValueError('record count')
    ids=np.frombuffer(take(4*U),dtype='<u4');sig=np.frombuffer(take(32*U),dtype='u1').reshape(U,32);hashes=np.frombuffer(take(32*U),dtype='u1').reshape(U,32);vectors=np.frombuffer(take(U*d*4),dtype='<f4').reshape(U,d)
    nr=u32()
    if nr>k:raise ValueError('result count')
    result=np.frombuffer(take(4*nr),dtype='<u4').tolist()
    if pos!=len(wire):raise ValueError('trailing bytes')
    return dict(n=n,bands=bands,width=width,k=k,L=L,radius=radius,query_hash=qh,mapping=mapping,leaves=leaves,proof=proof,ids=ids,signatures=sig,hashes=hashes,vectors=vectors,result=result)

def verify(wire,q,anchor,projection,k=20,radius=0):
    total=time.perf_counter();tim={};t=time.perf_counter();r=decode(wire);tim['deserialize_seconds']=time.perf_counter()-t
    if (r['n'],r['bands'],r['width'],r['L'],r['k'],r['radius'],r['query_hash'])!=(anchor['n'],anchor['bands'],anchor['width'],anchor['L'],k,radius,vh(q)):raise ValueError('request binding')
    t=time.perf_counter();qs=signature(q,projection);tim['query_signature_seconds']=time.perf_counter()-t
    t=time.perf_counter();keys=bandkeys(qs,r['bands'],r['width'],radius)
    if [m[0] for m in r['mapping']]!=keys:raise ValueError('mapped buckets differ')
    tim['mapped_bucket_recomputation_seconds']=time.perf_counter()-t
    t=time.perf_counter();bypos={x[0]:x for x in r['leaves']};required=set();full=set()
    if len(bypos)!=len(r['leaves']):raise ValueError('duplicate leaves')
    for key,lo,hi in r['mapping']:
        if lo==hi:
            if not 0<=lo<r['L'] or lo not in bypos or bypos[lo][1]!=key:raise ValueError('mapped leaf')
            required.add(lo);full.add(lo)
        else:
            if hi!=lo+1 or lo< -1 or hi>r['L']:raise ValueError('absence adjacency')
            if lo>=0:
                if lo not in bypos or not bypos[lo][1]<key:raise ValueError('absence lower bound')
                required.add(lo)
            if hi<r['L']:
                if hi not in bypos or not key<bypos[hi][1]:raise ValueError('absence upper bound')
                required.add(hi)
    if set(bypos)!=required or {i for i,x in bypos.items() if x[4] is not None}!=full:raise ValueError('disclosure scope')
    groups=[]
    for i in full:
        ids=bypos[i][4]
        if len(ids)!=bypos[i][2] or np.any(ids>=r['n']) or np.any(ids[1:]<=ids[:-1]):raise ValueError('bucket members')
        groups.append(ids)
    union=np.unique(np.concatenate(groups)) if groups else np.empty(0,dtype='<u4')
    if not np.array_equal(union,r['ids']):raise ValueError('candidate dictionary incomplete')
    tim['bucket_completeness_seconds']=time.perf_counter()-t
    t=time.perf_counter();commits={}
    for j,rid in enumerate(r['ids']):
        if vh(r['vectors'][j])!=bytes(r['hashes'][j]):raise ValueError('vector hash')
        commits[int(rid)]=rh(rid,r['signatures'][j],r['hashes'][j])
    tim['record_hash_verification_seconds']=time.perf_counter()-t
    t=time.perf_counter();leafhash={}
    for i,key,count,digest,ids in r['leaves']:
        if ids is not None:
            computed=H(b'M'+b''.join(commits[int(rid)] for rid in ids))
            if computed!=digest:raise ValueError('member digest')
        leafhash[i]=leaf(key,count,digest)
    if recover_root(leafhash,r['proof'],r['L'])!=anchor['root']:raise ValueError('MBT root mismatch')
    tim['mbt_root_reconstruction_seconds']=time.perf_counter()-t
    t=time.perf_counter();s=scores(r['vectors'],q);tim['cosine_recomputation_seconds']=time.perf_counter()-t
    t=time.perf_counter();expected=ranking(r['ids'],s,k);tim['topk_rerank_seconds']=time.perf_counter()-t
    t=time.perf_counter()
    if r['result']!=expected:raise ValueError('result mismatch')
    tim['result_comparison_seconds']=time.perf_counter()-t;tim['total_verification_seconds']=time.perf_counter()-total
    return tim

def anchor(index):return dict(n=index.n,bands=index.bands,width=index.width,L=index.tree.L,root=index.tree.root)

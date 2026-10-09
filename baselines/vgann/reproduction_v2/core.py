"""TrustMR experiments: reference V-GANN reproduction for baseline validation."""
import hashlib,heapq,struct,time
import numpy as np
import msgpack
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

U32=struct.Struct('<I'); U16=struct.Struct('<H'); MAGIC=b'VGANN002'
def sha(x):return hashlib.sha256(x).digest()
def parent(a,b):return sha(b'I'+a+b)
def leaf(rid,vector,tuples):return sha(b'L'+U32.pack(rid)+vector+U32.pack(len(tuples))+tuples)
def pack_tuple(t):return msgpack.packb(t,use_bin_type=True)
def unpack_tuple(b):return msgpack.unpackb(b,raw=False)
def initial(query_id,n):return int.from_bytes(sha(b'VGANN-screen-v1'+query_id.encode()),'big')%n


def guided_tuple(points,rid,neighbors):
    """Algorithm1, both recursive branches; stop when either side is empty."""
    p=points[rid];out=[]
    def rec(ids,code,d):
        if d>=len(p):out.append([code,list(map(int,ids))]);return
        neg=[int(i) for i in ids if points[i,d]<p[d]]
        pos=[int(i) for i in ids if points[i,d]>=p[d]]
        if not neg or not pos:out.append([code,list(map(int,ids))]);return
        rec(neg,code+'0',d+1);rec(pos,code+'1',d+1)
    rec(sorted(map(int,neighbors)),'',0)
    return out


def subspace(t,q,p):
    # Prefix-free codes identify one matching leaf without unused dimensions.
    best=None;score=-1
    for code,neighbors in t:
        n=0
        for j,c in enumerate(code):
            if c != ('1' if q[j]>p[j] else '0'):break
            n+=1
        if n>score:best=neighbors;score=n
    if best is None:raise ValueError('empty guided tuple')
    return best


def search(points,tuples,q,k,start):
    q=np.asarray(q,dtype=np.float64);distances={};candidate=[];visited={start};results={}
    def distance(i):
        if i not in distances:
            delta=np.asarray(points[i],dtype=np.float64)-q
            distances[i]=float(np.dot(delta,delta))
        return distances[i]
    ds=distance(start);heapq.heappush(candidate,(ds,start));results[start]=ds
    current=start;maxqueue=1
    while candidate:
        for rid in subspace(tuples[current],q,points[current]):
            if rid not in visited:
                visited.add(rid);heapq.heappush(candidate,(distance(rid),rid))
        maxqueue=max(maxqueue,len(candidate))
        d,current=heapq.heappop(candidate)
        worst=max(results,key=lambda i:(results[i],i))
        if len(results)==k and d>results[worst]:break
        results[current]=d
        if len(results)>k:
            worst=max(results,key=lambda i:(results[i],i));del results[worst]
    result=sorted(results,key=lambda i:(results[i],i))
    return result,visited,{'ndc':len(distances),'candidate_queue_max':maxqueue,'probed_points':len(visited)}


class Index:
    def __init__(self,points,graph,directory):
        from pathlib import Path
        self.points=points;self.n,self.d=points.shape;self.directory=Path(directory)
        t=time.perf_counter();self.tuples=[guided_tuple(points,i,graph[i]) for i in range(self.n)]
        self.tuplebytes=[pack_tuple(t) for t in self.tuples];self.tuple_seconds=time.perf_counter()-t
        t=time.perf_counter();self.size=1<<(self.n-1).bit_length()
        leaves=[leaf(i,points[i].astype('<f4',copy=False).tobytes(),self.tuplebytes[i]) for i in range(self.n)]
        leaves.extend(sha(b'P'+U32.pack(i)) for i in range(self.n,self.size));self.levels=[leaves]
        while len(leaves)>1:
            leaves=[parent(leaves[i],leaves[i+1]) for i in range(0,len(leaves),2)];self.levels.append(leaves)
        self.root=leaves[0];self.key=Ed25519PrivateKey.generate();self.public=self.key.public_key();self.signature=self.key.sign(self.root)
        self.mht_seconds=time.perf_counter()-t

    def proof(self,visited):
        t=time.perf_counter();entries=[]
        for i in range(self.size):
            if i in visited:
                entries.append((0,i,[self.points[i].astype('<f4',copy=False).tobytes(),self.tuplebytes[i]]))
            else:entries.append((0,i,self.levels[0][i]))
        rawseconds=time.perf_counter()-t;rawcount=len(entries);t=time.perf_counter();merges=0
        for level in range(len(self.levels)-1):
            new=[];j=0
            while j<len(entries):
                a=entries[j]
                if j+1<len(entries):
                    b=entries[j+1]
                    if a[0]==b[0]==level and isinstance(a[2],bytes) and isinstance(b[2],bytes) and a[1]%(1<<(level+1))==0 and b[1]==a[1]+(1<<level):
                        new.append((level+1,a[1],parent(a[2],b[2])));j+=2;merges+=1;continue
                new.append(a);j+=1
            entries=new
        return entries,{'raw_vo_seconds':rawseconds,'collapse_seconds':time.perf_counter()-t,'raw_vo_entries':rawcount,'hash_merges':merges,'full_entries':sum(not isinstance(x[2],bytes) for x in entries),'hash_entries':sum(isinstance(x[2],bytes) for x in entries),'vo_entries':len(entries)}

    def response(self,q,k,query_id):
        start=initial(query_id,self.n);t=time.perf_counter();result,visited,stats=search(self.points,self.tuples,q,k,start);stats['search_seconds']=time.perf_counter()-t
        entries,ps=self.proof(visited);stats.update(ps)
        response={'n':self.n,'d':self.d,'k':k,'start':start,'query_id':query_id,'entries':entries,'result':result,'signature':self.signature}
        t=time.perf_counter();payload,parts=encode(response);stats['serialize_seconds']=time.perf_counter()-t
        return response,payload,stats,parts


def encode(response):
    chunks=[];parts={k:0 for k in ['probed_point_vectors','guided_tuples','record_ids','merkle_hashes','structural_framing','initial_point_metadata','result_ids','result_vectors','root_signature']}
    def put(data,key):chunks.append(data);parts[key]+=len(data)
    qid=response['query_id'].encode()
    put(MAGIC+struct.pack('<IIIIH',response['n'],response['d'],response['k'],response['start'],len(qid))+qid,'initial_point_metadata')
    put(response['signature'],'root_signature');put(U32.pack(len(response['entries'])),'structural_framing')
    for level,rid,data in response['entries']:
        if isinstance(data,bytes):
            put(b'H'+U16.pack(level),'structural_framing');put(data,'merkle_hashes')
        else:
            vec,tup=data
            put(b'F'+U16.pack(level),'structural_framing');put(U32.pack(rid),'record_ids');put(vec,'probed_point_vectors');put(U32.pack(len(tup)),'structural_framing');put(tup,'guided_tuples')
    put(U32.pack(len(response['result']))+b''.join(U32.pack(i) for i in response['result']),'result_ids')
    payload=b''.join(chunks);assert len(payload)==sum(parts.values())
    parts['result_plus_proof_bytes']=len(payload);parts['proof_only_bytes']=len(payload)-parts['result_ids']
    return payload,parts


def decode(payload):
    offset=0
    def take(n):
        nonlocal offset
        if n<0 or offset+n>len(payload):raise ValueError('truncated response')
        v=payload[offset:offset+n];offset+=n;return v
    if take(8)!=MAGIC:raise ValueError('wire version')
    n,d,k,start,qlen=struct.unpack('<IIIIH',take(18));qid=take(qlen).decode();signature=take(64)
    if not 0<n<=100000000 or not 0<d<=100000 or not 0<k<=n:raise ValueError('wire dimensions')
    count=U32.unpack(take(4))[0]
    if count>2*(1<<(n-1).bit_length()):raise ValueError('entry count')
    entries=[];cursor=0
    for _ in range(count):
        tag=take(1);level=U16.unpack(take(2))[0]
        if level>(n-1).bit_length():raise ValueError('invalid level')
        if tag==b'H':data=take(32);rid=cursor
        elif tag==b'F':
            rid=U32.unpack(take(4))[0];data=[take(d*4),take(U32.unpack(take(4))[0])]
            if level!=0 or rid!=cursor or rid>=n:raise ValueError('record position')
        else:raise ValueError('entry tag')
        entries.append((level,rid,data));cursor+=1<<level
    count=U32.unpack(take(4))[0]
    if count>n:raise ValueError('result count')
    result=[U32.unpack(take(4))[0] for _ in range(count)]
    if offset!=len(payload):raise ValueError('trailing wire bytes')
    return {'n':n,'d':d,'k':k,'start':start,'query_id':qid,'signature':signature,'entries':entries,'result':result}


def root_hash(entries,n,d):
    stack=[];cursor=0
    for level,rid,data in entries:
        width=1<<level
        if rid!=cursor or cursor%width:raise ValueError('Merkle alignment')
        if isinstance(data,bytes):
            if len(data)!=32:raise ValueError('hash width')
            h=data
        else:
            if level!=0 or rid>=n or len(data[0])!=4*d:raise ValueError('point width')
            h=leaf(rid,*data)
        cursor+=width;stack.append((level,rid,h))
        while len(stack)>1:
            a,b=stack[-2:]
            if a[0]!=b[0] or a[1]%(1<<(a[0]+1)):break
            if b[1]!=a[1]+(1<<a[0]):raise ValueError('siblings')
            stack[-2:]=[(a[0]+1,a[1],parent(a[2],b[2]))]
    size=1<<(n-1).bit_length()
    if cursor!=size or len(stack)!=1 or stack[0][1]!=0:raise ValueError('incomplete Merkle coverage')
    return stack[0][2]


def verify(payload,q,k,query_id,n,d,public,trusted_root):
    times={};total=time.perf_counter();t=time.perf_counter();r=decode(payload);times['deserialize_seconds']=time.perf_counter()-t
    if (r['n'],r['d'],r['k'],r['query_id'],r['start'])!=(n,d,k,query_id,initial(query_id,n)):raise ValueError('request/start binding')
    t=time.perf_counter();root=root_hash(r['entries'],n,d);times['compute_root_seconds']=time.perf_counter()-t
    t=time.perf_counter()
    if root!=trusted_root:raise ValueError('trusted root mismatch')
    public.verify(r['signature'],root);times['signature_seconds']=time.perf_counter()-t
    if len(r['result'])!=k or len(set(r['result']))!=k:raise ValueError('result cardinality')
    t=time.perf_counter();points={};tuples={}
    for level,rid,data in r['entries']:
        if not isinstance(data,bytes):
            points[rid]=np.frombuffer(data[0],dtype='<f4');tuples[rid]=unpack_tuple(data[1])
    times['parse_seconds']=time.perf_counter()-t
    t=time.perf_counter();replayed,V,stats=search(points,tuples,q,k,r['start']);times['replay_seconds']=time.perf_counter()-t
    t=time.perf_counter()
    if replayed!=r['result']:raise ValueError('result replay mismatch')
    if V!=set(points):raise ValueError('declared probed set differs from replay')
    times['result_comparison_seconds']=time.perf_counter()-t;times['total_verification_seconds']=time.perf_counter()-total
    return times

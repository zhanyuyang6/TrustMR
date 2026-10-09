"""TrustMR experiments: independent V-GANN traversal checks against raw graph edges."""
from run import *

def reference(points,graph,q,k,start,variant,qid):
    n=len(points);q=np.asarray(q,dtype=np.float64);dist={};queue={};seen={start};R={};current=start;restarts=[]
    def distance(i):
        if i not in dist:dist[i]=float(sum((float(a)-float(b))**2 for a,b in zip(points[i],q)))
        return dist[i]
    def guided(i):
        ids=list(map(int,graph[i]));d=0
        while ids and d<points.shape[1]:
            neg=[j for j in ids if points[j,d]<points[i,d]];pos=[j for j in ids if points[j,d]>=points[i,d]]
            if not neg or not pos:break
            ids=pos if q[d]>points[i,d] else neg;d+=1
        return ids
    queue[start]=distance(start)
    while queue or (variant=='R3' and len(R)<k and len(seen)<n):
        if not queue:
            ids=sorted(set(map(int,graph[current]))-seen)
            if not ids:
                rid=int.from_bytes(hashlib.sha256(b'VGANN-R3'+qid.encode()+len(restarts).to_bytes(8,'little')).digest(),'big')%n
                while rid in seen:rid=(rid+1)%n
                restarts.append(rid);ids=[rid];current=rid
            for rid in ids:seen.add(rid);queue[rid]=distance(rid)
        ids=[i for i in guided(current) if i not in seen]
        if variant in ('R2','R3') and not ids and len(R)<k:ids=sorted(set(map(int,graph[current]))-seen)
        for rid in ids:seen.add(rid);queue[rid]=distance(rid)
        if not queue:break
        current=min(queue,key=lambda i:(queue[i],i));d=queue.pop(current)
        if len(R)==k and d>max(R.values()):break
        R[current]=d
        if len(R)>k:del R[max(R,key=lambda i:(R[i],i))]
    return sorted(R,key=lambda i:(R[i],i)),seen,restarts

def main():
    variant=json.loads((B/'selected_variant.json').read_text())['variant'].split('-')[-1]
    core=importlib.import_module('core_'+variant.lower());rows=[]
    for n in (1000,5000,20000,118287):
        points=np.load(GT/'corpus_vectors.npy',mmap_mode='r')[:n];graph=read_graph(OUT/f'graph_{n}.ivecs',n)
        tuples=[core.guided_tuple(points,i,graph[i]) for i in range(n)]
        for m,i,qid,q,gt in queries():
            rr,vv,restarts=reference(points,graph,q,20,core.initial(qid,n),variant,qid)
            r,v,s=core.search(points,tuples,q,20,core.initial(qid,n),qid,n)
            ok=(rr==r and vv==v and restarts==s['restart_points'])
            rows.append({'N':n,'query_id':qid,'match':ok});assert ok,(n,qid)
        print('REFERENCE PASS',n,flush=True)
    save('independent_reference.json',{'passed':True,'queries':len(rows),'checks':rows,'distance':'independent Python scalar sum of float64 squared differences','graph_neighbors':'raw graph recursive split, no authenticated tuple lookup'})
if __name__=='__main__':main()

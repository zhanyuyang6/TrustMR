"""TrustMR experiments: validate saved V-GANN results, index state, and wire sizes."""
from run import *
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

def load_index(core,n=118287):
    ix=core.Index.__new__(core.Index);ix.n=n;ix.d=1024;ix.points=np.load(GT/'corpus_vectors.npy',mmap_mode='r')[:n];ix.size=1<<(n-1).bit_length()
    data=(OUT/f'tuples_{n}.bin').read_bytes();pos=0;ix.tuplebytes=[]
    while pos<len(data):
        length=core.U32.unpack(data[pos:pos+4])[0];pos+=4;ix.tuplebytes.append(data[pos:pos+length]);pos+=length
    assert pos==len(data) and len(ix.tuplebytes)==n;ix.tuples=[core.unpack_tuple(t) for t in ix.tuplebytes]
    data=(OUT/f'mht_{n}.bin').read_bytes();pos=0;width=ix.size;ix.levels=[]
    while width:
        ix.levels.append([data[j:j+32] for j in range(pos,pos+width*32,32)]);pos+=width*32;width//=2
    assert pos==len(data);a=json.loads((OUT/f'anchor_{n}.json').read_text());ix.root=bytes.fromhex(a['root']);ix.signature=bytes.fromhex(a['signature']);ix.public=Ed25519PublicKey.from_public_bytes(bytes.fromhex(a['public_key']));assert ix.levels[-1][0]==ix.root
    return ix

def main():
    selected=json.loads((B/'selected_variant.json').read_text())['variant'].split('-')[-1];core=importlib.import_module('core_'+selected.lower());ix=load_index(core);results=[];reruns=[]
    for m,i,qid,q,gt in queries(100):
        r,V,s=core.search(ix.points,ix.tuples,q,20,core.initial(qid,ix.n),qid,ix.n)
        assert len(r)==20
        results.append({'modality':m,'query_index':i,'query_id':qid,'result_ids':r,'probed_ids':sorted(V),'restart_ids':s['restart_points']})
        if i==0:
            response,payload,stats,parts=ix.response(q,20,qid)
            assert payload==(OUT/f'example_{m}.bin').read_bytes();core.verify(payload,q,20,qid,ix.n,ix.d,ix.public,ix.root)
            reruns.append({'modality':m,'sha256':hashlib.sha256(payload).hexdigest(),'same_bytes':True})
    save('vgann_result_and_probed_ids.json',results)
    measured=list(csv.DictReader((OUT/'vgann_raw.csv').open()))
    for row,answer in zip(measured,results):
        assert row['query_id']==answer['query_id'];gt=np.load(GT/f"{row['modality']}_gt_ids.npy",mmap_mode='r')[int(row['query_index'])]
        assert float(row['recall'])==len(set(answer['result_ids'])&set(map(int,gt)))/20
        assert int(row['probed_points'])==len(answer['probed_ids'])
        assert int(row['ndc'])==int(row['probed_points'])
        parts=['probed_point_vectors','guided_tuples','record_ids','merkle_hashes','structural_framing','initial_point_metadata','restart_metadata','root_signature','result_ids','result_vectors']
        assert sum(int(row[k]) for k in parts)==int(row['result_plus_proof_bytes'])
        assert int(row['result_plus_proof_bytes'])-int(row['result_ids'])==int(row['proof_only_bytes'])
    assert (B/'faithful_v0.py').read_bytes()==(B.parent/'reproduction_v2/core.py').read_bytes()
    save('sanity.json',{'passed':True,'saved_index_reload':True,'all_200_result_recall_and_probed_counts_reproduced':True,'wire_breakdown_exact':True,'faithful_v0_byte_identical':True,'example_wire_checksums':reruns,'original_GT_reused':True})
    print('VALIDATION PASS',flush=True)
if __name__=='__main__':main()

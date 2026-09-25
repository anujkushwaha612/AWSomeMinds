import csv, re, random, collections, sys
sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(10**9)
import os; D=os.environ.get("BER_DATA", os.path.join(os.path.dirname(__file__), "..", "data", "dataset")) + "/"
dev=re.compile(r'[\u0900-\u097F]'); tok=re.compile(r'\w+')
def load(p):
    out={}
    with open(p,encoding='utf-8',newline='') as f:
        r=csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE); next(r)
        for row in r: out[row[0]]=(row[1],row[2],row[3])
    return out
# test side
t1=load(D+"test/test_source1.tsv")
for c in ("US","India","France"):
    n=sum(1 for v in t1.values() if v[2]==c); d=sum(1 for v in t1.values() if v[2]==c and dev.search(v[0]))
    print("test S1",c,n,"dev",d)
for fn in ("test_source2","test_source3"):
    t=load(D+f"test/{fn}.tsv"); cc=collections.Counter(v[2] for v in t.values())
    dv=collections.Counter(v[2] for v in t.values() if dev.search(v[0]))
    print(fn,dict(cc),"dev",dict(dv)); del t
s1=load(D+"train/train_source1.tsv"); s2=load(D+"train/train_source2.tsv"); s3=load(D+"train/train_source3.tsv")
gt={}
with open(D+"train/train_ground_truth.tsv",encoding='utf-8') as f:
    r=csv.reader(f,delimiter='\t'); next(r)
    for a,b in r: gt[a]=[x for x in b.split(',') if x]
def J(a,b):
    A=set(tok.findall(a.lower()));B=set(tok.findall(b.lower())); return len(A&B)/max(len(A|B),1)
random.seed(1); ex=[]; exdev=[]; addrJ=collections.defaultdict(list)
for a,v in gt.items():
    n1,a1,c=s1[a]
    for x in v:
        n2,a2,_=(s2 if x[1]=='2' else s3)[x]
        j=J(n1,n2); d=bool(dev.search(n2))
        if c=="India" and j<0.25:
            addrJ[("dev" if d else "latin", x[:2])].append(J(a1,a2))
            if not d and random.random()<0.0003 and len(ex)<20: ex.append((x[:2],n1,n2,a1,a2))
            if d and random.random()<0.0002 and len(exdev)<6: exdev.append((x[:2],n1,n2,a1,a2))
        if c=="US" and j<0.25:
            addrJ[("US",x[:2])].append(J(a1,a2))
for k,v in addrJ.items():
    v.sort(); n=len(v); print("addrJ lowNameJ",k,n,"median %.2f  p10 %.2f  share addrJ<0.25 %.3f"%(v[n//2],v[n//10],sum(1 for z in v if z<0.25)/n))
print("--- India low-J Latin examples (src | S1 name | partner name | S1 addr | partner addr)")
for e in ex: print(" | ".join(e))
print("--- Devanagari examples")
for e in exdev: print(" | ".join(e))

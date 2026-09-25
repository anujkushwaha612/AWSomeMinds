import csv, sys, re, random, collections
csv.field_size_limit(10**9)
import os; D=os.environ.get("BER_DATA", os.path.join(os.path.dirname(__file__), "..", "data", "dataset")) + "/"
dev=re.compile(r'[\u0900-\u097F]')
tok=re.compile(r'\w+')
def load(p):
    out={}
    with open(p,encoding='utf-8',newline='') as f:
        r=csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE); next(r)
        for row in r: out[row[0]]=(row[1],row[2],row[3])
    return out
s1=load(D+"train/train_source1.tsv")
s2=load(D+"train/train_source2.tsv"); s3=load(D+"train/train_source3.tsv")
gt={}
with open(D+"train/train_ground_truth.tsv",encoding='utf-8') as f:
    r=csv.reader(f,delimiter='\t'); next(r)
    for a,b in r: gt[a]=[x for x in b.split(',') if x]
matched=set(x for v in gt.values() for x in v)
for nm,s in (("S2",s2),("S3",s3)):
    orph=[k for k in s if k not in matched]
    byc=collections.Counter(s[k][2] for k in orph)
    print(nm,"records",len(s),"orphans",len(orph),f"{len(orph)/len(s):.3%}",dict(byc))
# per-source max / distribution
c2=collections.Counter(); c3=collections.Counter()
for a,v in gt.items():
    c2[sum(x.startswith('S2') for x in v)]+=1; c3[sum(x.startswith('S3') for x in v)]+=1
print("per-S1 S2-count dist",sorted(c2.items())); print("per-S1 S3-count dist",sorted(c3.items()))
# devanagari
def devshare(s,country):
    n=d=0
    for k,(nme,adr,c) in s.items():
        if c==country: n+=1; d+= bool(dev.search(nme))
    return d/max(n,1), n
for nm,s in (("S1",s1),("S2",s2),("S3",s3)):
    print("train",nm,"India dev-name share %.4f n=%d"%devshare(s,"India"))
def J(a,b):
    A=set(tok.findall(a.lower()));B=set(tok.findall(b.lower()))
    return len(A&B)/max(len(A|B),1)
# India pairs: low jaccard vs devanagari
stats=collections.Counter(); ex=[]
random.seed(0)
for a,v in gt.items():
    n1,a1,c=s1[a]
    if c!="India": continue
    for x in v:
        src=s2 if x.startswith('S2') else s3
        n2,a2,_=src[x]
        low=J(n1,n2)<0.25; d=bool(dev.search(n2))
        stats[(x[:2],low,d)]+=1
        if low and not d and random.random()<0.0005 and len(ex)<25: ex.append((n1,n2,a1,a2))
print("India pairs (src,lowJ,dev):",sorted(stats.items()))
for e in ex: print(" | ".join(e))
# address jaccard for dev pairs

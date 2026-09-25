import csv, re, collections, sys, unicodedata
sys.stdout.reconfigure(encoding='utf-8'); csv.field_size_limit(10**9)
import os; D=os.environ.get("BER_DATA", os.path.join(os.path.dirname(__file__), "..", "data", "dataset")) + "/"
blocks=[(0x900,'Deva'),(0x980,'Beng'),(0xA00,'Guru'),(0xA80,'Gujr'),(0xB00,'Orya'),(0xB80,'Taml'),(0xC00,'Telu'),(0xC80,'Knda'),(0xD00,'Mlym')]
def script(s):
    for ch in s:
        o=ord(ch)
        if 0x900<=o<0xD80: return blocks[(o-0x900)//0x80][1]
    return 'Latn'
dom=re.compile(r'\.(com|in|net|org|co)\b',re.I)
def norm(s): return ' '.join(re.findall(r'\w+',s.lower()))
def load(p):
    out={}
    with open(p,encoding='utf-8',newline='') as f:
        r=csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE); next(r)
        for row in r: out[row[0]]=(row[1],row[2],row[3])
    return out
for split in ("train","test"):
    for s in ("2","3"):
        t=load(D+f"{split}/{split}_source{s}.tsv")
        sc=collections.Counter(); dm=collections.Counter(); n=collections.Counter()
        for nm,ad,c in t.values():
            n[c]+=1
            if c=="India": sc[script(nm)]+=1
            if dom.search(nm): dm[c]+=1
        tot=n["India"]
        print(split,"S"+s,"India name scripts:",{k:f"{v/tot:.3%}" for k,v in sc.most_common()})
        print(split,"S"+s,"domain-style names:",{c:f"{dm[c]/n[c]:.2%}" for c in n})
        if split=="train":
            if s=="2": T2=t
            else: T3=t
        else: del t
s1=load(D+"train/train_source1.tsv")
matched=set()
with open(D+"train/train_ground_truth.tsv",encoding='utf-8') as f:
    r=csv.reader(f,delimiter='\t'); next(r)
    for a,b in r: matched.update(x for x in b.split(',') if x)
s1names=collections.Counter((norm(v[0]),v[2]) for v in s1.values())
for nm,T in (("S2",T2),("S3",T3)):
    h=collections.Counter(); n=collections.Counter()
    for k,(a,b,c) in T.items():
        key='matched' if k in matched else 'orphan'
        n[key]+=1; h[key]+= (norm(a),c) in s1names
    print(nm,"exact-norm-name hit in S1 (same country):",{k:f"{h[k]/n[k]:.2%}" for k in n})

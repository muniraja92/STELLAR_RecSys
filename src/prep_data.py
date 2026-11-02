"""Leakage-free data preparation for STELLAR-KG (revision).

Produces, per dataset, a compressed npz/pickle bundle with
  - chronologically ordered user sequences (item ids, timestamps)
  - leave-one-out split: last item -> test, second-to-last -> validation, rest -> train
  - item-side knowledge graph triples (item, relation, entity) built ONLY from item metadata
    (no behaviour-derived relations, so no information from validation/test interactions can
    enter the graph).
"""
import os, sys, json, pickle, collections
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get('STELLAR_RAW', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'raw'))
OUT = os.path.join(ROOT, 'data')


def kcore(df, ucol, icol, umin, imin):
    while True:
        n0 = len(df)
        uc = df[ucol].value_counts(); df = df[df[ucol].isin(uc[uc >= umin].index)]
        ic = df[icol].value_counts(); df = df[df[icol].isin(ic[ic >= imin].index)]
        if len(df) == n0:
            return df


def finalize(name, df, item_meta_triples, item_key_to_raw):
    """df: columns user_raw, item_raw, ts, order (stable tie-break). triples: list of (item_raw, rel, ent_str)."""
    df = df.sort_values(['user_raw', 'ts', 'order'], kind='mergesort')
    users = df['user_raw'].unique(); items = np.sort(df['item_raw'].unique())
    u2i = {u: k for k, u in enumerate(users)}; it2i = {v: k for k, v in enumerate(items)}
    df = df.assign(u=df['user_raw'].map(u2i).values, i=df['item_raw'].map(it2i).values)
    seqs, tss = [], []
    for u, g in df.groupby('u', sort=True):
        seqs.append(g['i'].to_numpy(np.int64)); tss.append(g['ts'].to_numpy(np.int64))
    n_users, n_items = len(users), len(items)
    # split
    train = [s[:-2] for s in seqs]; val = np.array([s[-2] for s in seqs]); test = np.array([s[-1] for s in seqs])
    # KG: map item_raw -> idx, entities -> ids; drop entities linked to < 2 items (carry no shared signal)
    trip = [(it2i[a], r, e) for (a, r, e) in item_meta_triples if a in it2i]
    trip = list(dict.fromkeys(trip))
    ent_items = collections.defaultdict(set)
    for a, r, e in trip:
        ent_items[(r, e)].add(a)
    trip = [(a, r, e) for (a, r, e) in trip if len(ent_items[(r, e)]) >= 2]
    rels = sorted({r for _, r, _ in trip}); r2i = {r: k for k, r in enumerate(rels)}
    ents = sorted({(r, e) for _, r, e in trip}); e2i = {x: k for k, x in enumerate(ents)}
    kg = np.array([(a, r2i[r], e2i[(r, e)]) for a, r, e in trip], dtype=np.int64).reshape(-1, 3)
    bundle = dict(name=name, n_users=n_users, n_items=n_items, seqs=seqs, ts=tss,
                  train=train, val=val, test=test, kg=kg, relations=rels, n_entities=len(ents),
                  item_raw=items, user_raw=users)
    with open(os.path.join(OUT, f'{name}.pkl'), 'wb') as f:
        pickle.dump(bundle, f)
    n_int = sum(len(s) for s in seqs)
    stats = dict(dataset=name, users=n_users, items=n_items, interactions=int(n_int),
                 train_interactions=int(sum(len(s) for s in train)),
                 avg_seq_len=float(n_int / n_users), density=float(n_int / (n_users * n_items)),
                 kg_triples=int(len(kg)), kg_relations=len(rels), kg_entities=len(ents),
                 items_with_kg=int(len(np.unique(kg[:, 0])) if len(kg) else 0),
                 relation_counts={r: int((kg[:, 1] == r2i[r]).sum()) for r in rels})
    with open(os.path.join(OUT, f'{name}_stats.json'), 'w') as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


def prep_ml1m():
    if os.path.exists(f'{RAW}/ml1m/ratings.dat'):      # official GroupLens release
        r = pd.read_csv(f'{RAW}/ml1m/ratings.dat', sep='::', engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
        mv = pd.read_csv(f'{RAW}/ml1m/movies.dat', sep='::', engine='python', names=['movieId', 'title', 'genres'], encoding='latin-1')
    else:                                              # identical content mirrored as CSV
        r = pd.read_csv(f'{RAW}/ml1m/ratings.csv')
        r['timestamp'] = r['timestamp'].astype(str).str.strip().astype(np.int64)
        mv = pd.read_csv(f'{RAW}/ml1m/movies.csv')
    assert len(r) == 1000209 and r.userId.nunique() == 6040 and r.movieId.nunique() == 3706
    r['order'] = np.arange(len(r))
    r = r[r.rating >= 4]
    r = kcore(r, 'userId', 'movieId', 10, 5)
    df = r.rename(columns={'userId': 'user_raw', 'movieId': 'item_raw', 'timestamp': 'ts'})[['user_raw', 'item_raw', 'ts', 'order']]
    triples = []
    for _, row in mv.iterrows():
        for g in str(row['genres']).split('|'):
            triples.append((row['movieId'], 'genre(MovieLens)', g))
        yr = str(row['title']).strip()[-5:-1]
        if yr.isdigit():
            triples.append((row['movieId'], 'release_decade', yr[:3] + '0s'))
    # Satori KG (RippleNet release): keep 1-hop item-attribute triples
    m = pd.read_csv(f'{RAW}/ripplenet/item_index2entity_id_rehashed.txt', sep='\t', header=None, names=['movieId', 'ent'])
    ent2movie = dict(zip(m.ent, m.movieId))
    kg = pd.read_csv(f'{RAW}/ripplenet/kg_all.txt', sep='\t', header=None, names=['h', 'r', 't'])
    keep = {'film.film.genre': 'genre', 'film.film.language': 'language', 'film.film.director': 'director',
            'film.film.writer': 'writer', 'film.film.star': 'star', 'film.film.country': 'country',
            'film.film.rating': 'rating', 'film.director.film': 'director', 'film.writer.film': 'writer',
            'film.actor.film': 'actor', 'film.person_or_entity_appearing_in_film.film': 'actor',
            'film.genre.film': 'genre'}
    hm = kg[kg.h.isin(ent2movie) & kg.r.isin(keep)]
    for h, rr, t in hm.itertuples(index=False):
        triples.append((ent2movie[h], keep[rr], f'e{t}'))
    tm = kg[kg.t.isin(ent2movie) & kg.r.isin(keep) & ~kg.h.isin(ent2movie)]
    for h, rr, t in tm.itertuples(index=False):
        triples.append((ent2movie[t], keep[rr], f'e{h}'))
    finalize('ml1m', df, triples, None)


def prep_amazon(name, folder, root_cat, meta_files):
    r = pd.concat([pd.read_parquet(f'{RAW}/{folder}/reviews_train-0000{k}-of-00002.parquet', columns=['reviewerID', 'reviewTime', 'asin']) for k in (0, 1)])
    r = r.explode(['reviewTime', 'asin'])
    r['ts'] = (pd.to_datetime(r['reviewTime']) - pd.Timestamp('1970-01-01')) // pd.Timedelta(seconds=1)   # unix seconds, independent of the datetime resolution
    r['order'] = np.arange(len(r))
    r = r.drop_duplicates(['reviewerID', 'asin'], keep='first')
    r = kcore(r, 'reviewerID', 'asin', 5, 5)
    df = r.rename(columns={'reviewerID': 'user_raw', 'asin': 'item_raw'})[['user_raw', 'item_raw', 'ts', 'order']]
    meta = pd.concat([pd.read_parquet(f'{RAW}/{folder}/{mf}', columns=['asin', 'brand', 'categories', 'price']) for mf in meta_files])
    meta = meta[meta.asin.isin(set(df.item_raw))]
    triples = []
    for asin, brand, cats, price in meta.itertuples(index=False):
        if isinstance(brand, str) and brand.strip():
            triples.append((asin, 'brand', brand.strip().lower()))
        if cats is not None:
            for path in cats:
                path = [c for c in list(path) if c]
                for depth, c in enumerate(path):
                    if depth == 0 and c == root_cat:
                        continue
                    triples.append((asin, 'category', c))
        if price is not None and not (isinstance(price, float) and np.isnan(price)):
            p = float(price)
            b = '<5' if p < 5 else '5-10' if p < 10 else '10-20' if p < 20 else '20-40' if p < 40 else '40-80' if p < 80 else '80+'
            triples.append((asin, 'price_band', b))
    finalize(name, df, triples, None)


def download():
    """Fetch raw data (MovieLens-1M from GroupLens, Satori KG from the RippleNet release,
    Amazon 2014 Beauty / Toys reviews + metadata from a public Hugging Face mirror)."""
    import urllib.request, zipfile, io
    def get(url, dst):
        if os.path.exists(dst): return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print('downloading', url, flush=True)
        urllib.request.urlretrieve(url, dst)
    try:
        if not os.path.exists(f'{RAW}/ml1m/ratings.dat'):
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://files.grouplens.org/datasets/movielens/ml-1m.zip', timeout=120).read()))
            os.makedirs(f'{RAW}/ml1m', exist_ok=True)
            for n in ('ratings.dat', 'movies.dat'):
                open(f'{RAW}/ml1m/{n}', 'wb').write(z.read(f'ml-1m/{n}'))
    except Exception as e:     # MovieLens-1M may only be obtained from GroupLens (licence); no mirror is used
        raise SystemExit(f'Download MovieLens-1M from https://grouplens.org/datasets/movielens/1m/ into {RAW}/ml1m ({e})')
    rp = 'https://raw.githubusercontent.com/hwwang55/RippleNet/master/data/movie/'
    for f in ('item_index2entity_id_rehashed.txt', 'kg_part1_rehashed.txt', 'kg_part2_rehashed.txt'):
        get(rp + f, f'{RAW}/ripplenet/{f}')
    if not os.path.exists(f'{RAW}/ripplenet/kg_all.txt'):
        with open(f'{RAW}/ripplenet/kg_all.txt', 'w') as o:
            for f in ('kg_part1_rehashed.txt', 'kg_part2_rehashed.txt'):
                o.write(open(f'{RAW}/ripplenet/{f}').read())
    hf = 'https://huggingface.co/datasets/milistu/'
    for k in (0, 1):
        get(f'{hf}Amazon_Beauty_2014/resolve/main/reviews/train-0000{k}-of-00002.parquet', f'{RAW}/beauty/reviews_train-0000{k}-of-00002.parquet')
        get(f'{hf}Amazon_Toys_and_Games_2014/resolve/main/reviews/train-0000{k}-of-00002.parquet', f'{RAW}/toys/reviews_train-0000{k}-of-00002.parquet')
        get(f'{hf}Amazon_Toys_and_Games_2014/resolve/main/metadata/train-0000{k}-of-00002.parquet', f'{RAW}/toys/metadata_train-0000{k}-of-00002.parquet')
    get(f'{hf}Amazon_Beauty_2014/resolve/main/metadata/train-00000-of-00001.parquet', f'{RAW}/beauty/metadata_train-00000-of-00001.parquet')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if os.environ.get('STELLAR_DOWNLOAD'): download()
    which = sys.argv[1:] or ['ml1m', 'beauty', 'toys']
    if 'ml1m' in which: prep_ml1m()
    if 'beauty' in which: prep_amazon('beauty', 'beauty', 'Beauty', ['metadata_train-00000-of-00001.parquet'])
    if 'toys' in which: prep_amazon('toys', 'toys', 'Toys & Games', ['metadata_train-00000-of-00002.parquet', 'metadata_train-00001-of-00002.parquet'])

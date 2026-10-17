"""Minimal pure-Python Parquet reader (flat schema; PLAIN / dictionary pages;
UNCOMPRESSED / SNAPPY / GZIP / ZSTD-if-available). Enough for a geopandas file."""
import struct, zlib, sys
import numpy as np

# ---------------- thrift compact protocol ----------------
class TC:
    def __init__(self, b, p=0): self.b, self.p = b, p
    def byte(self):
        v = self.b[self.p]; self.p += 1; return v
    def varint(self):
        r = s = 0
        while True:
            x = self.byte(); r |= (x & 0x7f) << s; s += 7
            if not x & 0x80: return r
    def zz(self):
        v = self.varint(); return (v >> 1) ^ -(v & 1)
    def binary(self):
        n = self.varint(); v = self.b[self.p:self.p + n]; self.p += n; return v
    def value(self, t):
        if t in (1, 2): return t == 1
        if t == 3: return struct.unpack('b', bytes([self.byte()]))[0]
        if t in (4, 5, 6): return self.zz()
        if t == 7:
            v = struct.unpack('<d', self.b[self.p:self.p + 8])[0]; self.p += 8; return v
        if t == 8: return self.binary()
        if t in (9, 10):
            h = self.byte(); n = h >> 4; et = h & 0x0f
            if n == 15: n = self.varint()
            return [self.value(et) for _ in range(n)]
        if t == 11:
            n = self.varint()
            if n == 0: return {}
            kv = self.byte(); return {self.value(kv >> 4): self.value(kv & 0xf) for _ in range(n)}
        if t == 12: return self.struct()
        raise ValueError(f"thrift type {t}")
    def struct(self):
        out, last = {}, 0
        while True:
            h = self.byte()
            if h == 0: return out
            d, t = h >> 4, h & 0x0f
            fid = last + d if d else self.zz()
            last = fid
            out[fid] = self.value(t)

# ---------------- codecs ----------------
def snappy(b):
    p = 0; n = s = 0
    while True:
        x = b[p]; p += 1; n |= (x & 0x7f) << s; s += 7
        if not x & 0x80: break
    out = bytearray()
    L = len(b)
    while p < L:
        tag = b[p]; p += 1; t = tag & 3
        if t == 0:
            ln = tag >> 2
            if ln >= 60:
                k = ln - 59; ln = int.from_bytes(b[p:p + k], 'little'); p += k
            ln += 1; out += b[p:p + ln]; p += ln
        else:
            if t == 1:
                ln = ((tag >> 2) & 7) + 4; off = ((tag >> 5) << 8) | b[p]; p += 1
            elif t == 2:
                ln = (tag >> 2) + 1; off = int.from_bytes(b[p:p + 2], 'little'); p += 2
            else:
                ln = (tag >> 2) + 1; off = int.from_bytes(b[p:p + 4], 'little'); p += 4
            st = len(out) - off
            if off >= ln: out += out[st:st + ln]
            else:
                for i in range(ln): out.append(out[st + i])
    assert len(out) == n, (len(out), n)
    return bytes(out)

def decompress(codec, b):
    if codec == 0: return b
    if codec == 1: return snappy(b)
    if codec == 2: return zlib.decompress(b, 47)
    if codec == 6:
        import zstandard; return zstandard.ZstdDecompressor().decompress(b)
    raise ValueError(f"codec {codec}")

# ---------------- encodings ----------------
def rle_bitpacked(b, bw, count):
    out = []; p = 0; L = len(b)
    nbytes = (bw + 7) // 8
    while len(out) < count and p < L:
        h = 0; s = 0
        while True:
            x = b[p]; p += 1; h |= (x & 0x7f) << s; s += 7
            if not x & 0x80: break
        if h & 1:
            g = h >> 1; nb = g * bw
            chunk = int.from_bytes(b[p:p + nb], 'little'); p += nb
            mask = (1 << bw) - 1
            out.extend((chunk >> (i * bw)) & mask for i in range(g * 8))
        else:
            r = h >> 1; v = int.from_bytes(b[p:p + nbytes], 'little'); p += nbytes
            out.extend([v] * r)
    return out[:count]

PHYS = {1: '<i4', 2: '<i8', 4: '<f4', 5: '<f8'}

def plain(b, ptype, n):
    if ptype in PHYS:
        dt = np.dtype(PHYS[ptype]); return list(np.frombuffer(b, dt, n))
    if ptype == 6:
        out = []; p = 0
        for _ in range(n):
            ln = struct.unpack('<I', b[p:p + 4])[0]; p += 4; out.append(b[p:p + ln]); p += ln
        return out
    if ptype == 0:
        return [bool((b[i // 8] >> (i % 8)) & 1) for i in range(n)]
    raise ValueError(f"ptype {ptype}")

# ---------------- reader ----------------
def read(path, columns=None):
    b = open(path, 'rb').read()
    n = struct.unpack('<I', b[-8:-4])[0]
    meta = TC(b, len(b) - 8 - n).struct()
    schema = meta[2]
    leaves = [s for s in schema[1:] if 5 not in s or not s.get(5)]
    names = [s[4].decode() for s in leaves]
    optional = {s[4].decode(): s.get(3, 0) == 1 for s in leaves}
    want = set(columns or names)
    data = {c: [] for c in names if c in want}
    for rg in meta[4]:
        for cc in rg[1]:
            md = cc[3]
            name = b'.'.join(md[3]).decode()
            if name not in data: continue
            ptype, codec = md[1], md[4]
            nvals = md[5]
            start = md.get(11, md[9])                       # dictionary page offset first
            end = md[9] + md[7]
            if 11 in md: end = max(end, md[11] + md[7])
            p = start; dictionary = None; got = 0
            while got < nvals:
                tc = TC(b, p); ph = tc.struct(); p = tc.p
                csize, usize, ptyp = ph[3], ph[2], ph[1]
                raw = b[p:p + csize]; p += csize
                if ptyp == 2:                               # dictionary page
                    dh = ph[7]
                    dictionary = plain(decompress(codec, raw), ptype, dh[1]); continue
                if ptyp == 0:                               # data page v1
                    dh = ph[5]; nv = dh[1]; enc = dh[2]
                    page = decompress(codec, raw); q = 0
                    if optional[name]:
                        ln = struct.unpack('<I', page[q:q + 4])[0]; q += 4
                        defs = rle_bitpacked(page[q:q + ln], 1, nv); q += ln
                    else:
                        defs = [1] * nv
                    nn = sum(defs); body = page[q:]
                elif ptyp == 3:                             # data page v2
                    dh = ph[8]; nv = dh[1]; nnull = dh[2]; enc = dh[4]
                    dl, rl = dh[5], dh[6]
                    lv = raw[:dl + rl]; rest = raw[dl + rl:]
                    if dh.get(7, True): rest = decompress(codec, rest)
                    defs = rle_bitpacked(lv[rl:], 1, nv) if optional[name] and dl else [1] * nv
                    nn = nv - nnull; body = rest
                else:
                    continue
                if enc in (2, 8):                           # dictionary indices
                    bw = body[0]
                    idx = rle_bitpacked(body[1:], bw, nn)
                    vals = [dictionary[i] for i in idx]
                elif enc == 0:
                    vals = plain(body, ptype, nn)
                else:
                    raise ValueError(f"encoding {enc}")
                it = iter(vals)
                data[name].extend(next(it) if d else None for d in defs)
                got += nv
    return data, meta

if __name__ == "__main__":
    d, m = read(sys.argv[1])
    print({k: (len(v), v[:2]) for k, v in d.items()})

"""Seekable concatenation of AI Hub ZIP parts, without a merged copy."""
import bisect
import io
from pathlib import Path


class JoinedParts(io.RawIOBase):
    def __init__(self, parts):
        self.parts=sorted(map(Path,parts),key=lambda p:int(p.suffix[5:]))
        self.starts=[];self.pos=0;self.total=0;self.handle=None;self.index=-1
        if not self.parts: raise ValueError('No archive parts')
        for p in self.parts:
            if int(p.suffix[5:])!=self.total: raise ValueError('Missing ZIP part')
            self.starts.append(self.total);self.total+=p.stat().st_size
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.total+offset
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos
    def read(self,size=-1):
        size=self.total-self.pos if size<0 else min(size,self.total-self.pos);chunks=[]
        while size>0:
            idx=bisect.bisect_right(self.starts,self.pos)-1
            if idx!=self.index:
                if self.handle:self.handle.close()
                self.handle=self.parts[idx].open('rb');self.index=idx
            self.handle.seek(self.pos-self.starts[idx])
            end=self.starts[idx+1] if idx+1<len(self.starts) else self.total
            chunk=self.handle.read(min(size,end-self.pos))
            if not chunk:break
            chunks.append(chunk);self.pos+=len(chunk);size-=len(chunk)
        return b''.join(chunks)
    def close(self):
        if self.handle:self.handle.close()
        super().close()

"""Synchronized direct classification speed: no generation and no HTTP."""
import gc
import time
import torch
from experiment_common import quantiles


@torch.inference_mode()
def benchmark(step, tokenizer, reference_tokenizer):
    def ids(text):return tokenizer.encode(text,add_special_tokens=False)
    def ref_count(text):return len(reference_tokenizer.encode(text,add_special_tokens=False))
    def call(token_ids,cache=None,cached=False):
        risk,category,state=step(torch.tensor([token_ids],device='cuda'),cache,cached)
        # Include a CPU-visible classification result in the timing.
        risk.float().softmax(-1).cpu().tolist()
        category.float().softmax(-1).cpu().tolist()
        return risk.float().softmax(-1),state
    result={'scope':'L20 in-process, pre-tokenized, synchronized CPU-visible risk/category probabilities; no HTTP or batching',
            'generated_tokens':0,'complete_same_text':{},'stream_same_text':{}}
    phrase='公开资料与 public information。\n'
    for repeats in (16,128):
        text='USER:\n'+phrase*repeats; encoded=ids(text)
        for _ in range(2):call(encoded)
        times=[]
        for _ in range(10):
            torch.cuda.synchronize();start=time.perf_counter();call(encoded);torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
        total=sum(times)
        result['complete_same_text'][str(repeats)]={
            'characters_per_request':len(text),'native_tokens_per_request':len(encoded),
            'reference_tokens_per_request':ref_count(text),'requests':len(times),
            'native_itps':len(times)*len(encoded)/total,
            'reference_itps':len(times)*ref_count(text)/total,
            'classifications_per_second':len(times)/total,**quantiles(times)}
    for sessions in (1,8,32):
        prefix='USER:\n'+phrase*32
        pieces=[];previous=ids(prefix);current=prefix
        for _ in range(8):
            current+=phrase;updated=ids(current)
            if updated[:len(previous)]!=previous:
                raise RuntimeError('Benchmark text boundary is unstable for this tokenizer')
            pieces.append(updated[len(previous):]);previous=updated
        states=[]
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        before=torch.cuda.memory_allocated()
        for _ in range(sessions):
            _,state=call(ids(prefix),cached=True);states.append(state)
        torch.cuda.synchronize();state_bytes=torch.cuda.memory_allocated()-before
        times=[];native=0
        start_all=time.perf_counter()
        for piece in pieces:
            for index in range(sessions):
                torch.cuda.synchronize();started=time.perf_counter()
                p,state=call(piece,states[index],True);states[index]=state
                torch.cuda.synchronize();times.append(time.perf_counter()-started);native+=len(piece)
        wall=time.perf_counter()-start_all
        full,_=call(ids(current))
        error=float((full-p).abs().max())
        if error>.03:raise RuntimeError(f'Stream/full risk parity failed: {error}')
        result['stream_same_text'][str(sessions)]={
            'sessions':sessions,'classification_results':len(times),
            'logical_new_native_tokens':native,
            'logical_new_reference_tokens':sessions*(ref_count(current)-ref_count(prefix)),
            'native_itps':native/wall,
            'reference_itps':sessions*(ref_count(current)-ref_count(prefix))/wall,
            'classifications_per_second':len(times)/wall,
            'prefilled_cache_allocation_bytes':state_bytes,
            'full_vs_stream_risk_max_abs_error':error,**quantiles(times)}
        del states,state,p,full
        gc.collect();torch.cuda.empty_cache()
    return result

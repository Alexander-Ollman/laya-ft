"""Masked length buckets bound MPS graph shapes without dropping text."""
def pad_batch(batch,pad_id,multiple=128):
    if not isinstance(multiple,int) or isinstance(multiple,bool) or multiple<0:
        raise ValueError('padding multiple must be a nonnegative integer')
    if multiple==0:return batch
    import torch.nn.functional as F
    length=batch['input_ids'].shape[1];extra=(-length)%multiple
    if not extra:return batch
    out=dict(batch)
    out['input_ids']=F.pad(batch['input_ids'],(0,extra),value=pad_id)
    out['attention_mask']=F.pad(batch['attention_mask'],(0,extra),value=0)
    return out


def install_agent_padding(multiple=128):
    """Process-local adapter; preserves Laya's inference/decoding implementation."""
    import laya.agent as module
    from laya.common import collate_items
    original=module.collate_items
    module.collate_items=lambda batch,pad_id:pad_batch(collate_items(batch,pad_id),pad_id,multiple)
    return original

if __name__ == '__main__':

    from argparse import ArgumentParser
    import clip
    import torch
    from torch.utils.data.dataloader import DataLoader
    from egg.zoo.emergent_captioner.dataloaders.flickr_dataloader import get_dataloader

    parser = ArgumentParser()
    parser.add_argument("--output_file_prefix", required=True)
    parser.add_argument("--clip_model", choices="ViT-B/16 ViT-B/32".split(), default="ViT-B/32",)
    parser.add_argument("--dataset_dir", default="/checkpoint/rdessi/datasets/flickr30k/")
    parser.add_argument("--split", choices="train val".split(), default="train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=torch.device, default=torch.device('cpu'))

    opts = parser.parse_args()

    dataloader: DataLoader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        split=opts.split,
        num_workers=opts.num_workers
    )
    clip_model = clip.load(opts.clip_model)[0].eval().to(opts.device)

    emb_n = len(dataloader.dataset)
    emb_s = clip_model.text_projection.shape[1]
    prec_emb = torch.empty(emb_n, emb_s, dtype=torch.float32, device='cpu')

    i = 0
    for batch in dataloader:
        batch, *_ = batch
        batch = batch.to(opts.device)
        with torch.no_grad():
            feats = clip.encode_image(batch)
            feats /= feats.norm(dim=-1, keepdim=True)
            feats = feats.to('cpu')
            prec_emb[i:i + feats.size(0)] = feats
            i += feats.size(0)

    prec_nns = torch.empty(emb_n, 100, dtype=torch.int32, device='cpu')
    for chunk_start in range(0, emb_n, 1000):
        chunk = prec_emb[chunk_start:chunk_start + 1000]
        # emb_n x 1000
        cosin = prec_emb @ chunk.t()
        # 101 x 1000
        nns = torch.topk(cosin, k=100+1, dim=0, largest=True, sorted=True).indices
        # 1000 x 101
        prec_nns[chunk_start:chunk_start+1000] = nns

    torch.save(prec_emb, opts.output_file_prefix + '.emb.pt')
    torch.save(prec_nns, opts.output_file_prefix + '.nns.pt')

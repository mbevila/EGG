from egg.zoo.emergent_captioner.finetuning.sender import ClipCapSender
from argparse import ArgumentParser
import torch

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        "--clipcap_model_path",
        default="/private/home/rdessi/EGG/egg/zoo/emergent_captioner/clipclap_models/conceptual_weights.pt",
        )
    parser.add_argument(
        "--clip_model",
        choices=["ViT-B/16", "ViT-B/32"],
        default="ViT-B/32",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device('cpu'),
    )
    parser.add_argument("--dataset_path")
    parser.add_argument("--dataset_metadata_path")
    parser.add_argument("--dataset_format", choices="coco flickr".split(), default="coco")
    parser.add_argument("--dataset_split", choices="train val test".split(), default="val")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    model = ClipCapSender(
        nb_prefix_tokens=10,
        clip_model=args.clip_model,
        clipcap_path=args.clipcap_model_path,
        num_return_sequences=1,
        do_sample=False,
        beam_size=5,
        max_len=50,
    ).eval()
    model.clipcap.load_state_dict(torch.load(args.clipcap_model_path, map_location='cpu'))
    model.clipcap.maybe_patch_gpt(1000)
    model = model.to(args.device)

    if args.dataset_format == "coco":
        from egg.zoo.emergent_captioner.dataloaders.coco_dataloader import get_dataloader
        dataloader = get_dataloader(
            image_dir=args.dataset_path,
            metadata_dir=args.dataset_metadata_path,
            split=args.dataset_split,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
    elif args.dataset_format == "flickr":
        from egg.zoo.emergent_captioner.dataloaders.flickr_dataloader import get_dataloader

        dataloader = get_dataloader(
            dataset_dir=args.dataset_path,
            split=args.dataset_split,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
    else:
        raise NotImplementedError

    for batch in dataloader:
        images, *_ = batch
        images = images.to(args.device)
        captions = model(images)[0]
        captions = [c.strip() for c in captions]
        for c in captions:
            print(c)





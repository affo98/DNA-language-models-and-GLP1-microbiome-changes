import torch 

def get_optimizer(model, args):
    if args.distributed:
        optimizer = torch.optim.AdamW([
                {'params':[p for n, p in model.module.dnabert2.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'lr': args.lr, 'weight_decay': 0.01},
                {'params':[p for n, p in model.module.dnabert2.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'lr': args.lr, 'weight_decay': 0},
                {'params':[p for n, p in model.module.contrast_head.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'lr': args.lr*args.lr_scale, 'weight_decay': 0.01},
                {'params':[p for n, p in model.module.contrast_head.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'lr': args.lr*args.lr_scale, 'weight_decay': 0}
                ])
    else:
        optimizer = torch.optim.AdamW([
                {'params':[p for n, p in model.dnabert2.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'lr': args.lr, 'weight_decay': 0.01},
                {'params':[p for n, p in model.dnabert2.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'lr': args.lr, 'weight_decay': 0},
                {'params':[p for n, p in model.contrast_head.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'lr': args.lr*args.lr_scale, 'weight_decay': 0.01},
                {'params':[p for n, p in model.contrast_head.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight'])], 'lr': args.lr*args.lr_scale, 'weight_decay': 0}
                ])
    return optimizer 
    


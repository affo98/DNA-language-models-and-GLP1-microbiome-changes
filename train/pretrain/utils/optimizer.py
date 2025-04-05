import torch 

def get_optimizer(model, args):
    optimizer = torch.optim.AdamW([
            {'params':model.module.dnabert2.parameters(), 'weight_decay': args.weight_decay}, 
            {'params':model.module.contrast_head.parameters(), 'lr': args.lr*args.lr_scale, 'weight_decay': args.weight_decay*0.1}], lr=args.lr)
    return optimizer 
    


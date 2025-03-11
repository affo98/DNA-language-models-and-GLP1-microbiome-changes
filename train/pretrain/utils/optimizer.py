import torch 

def get_optimizer(model, args):
    optimizer = torch.optim.Adam([
            {'params':model.module.dnabert2.parameters()}, 
            {'params':model.module.contrast_head.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    return optimizer 
    


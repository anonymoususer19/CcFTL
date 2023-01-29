import torch

def global_communicate(model1, model2, dist, args):
    with torch.no_grad():
        for param in model1.parameters():
            if args.is_master:
                g_param = torch.zeros(param.shape)
                l_param = [torch.zeros(param.shape) for _ in range(args.world_size)]
            else:
                g_param = param.cpu()
                l_param = list()
                b_param = torch.zeros(param.shape)
            
            dist.gather(g_param, l_param, dst = args.master_rank)

            if args.is_master:
                b_param = sum(l_param) / (args.world_size-1)
            dist.broadcast(b_param, src = args.master_rank)

            if not args.is_master:
                b_param = b_param.to(args.device)
            param *= 0
            param.set_(b_param)

        for param in model2.parameters():
            if args.is_master:
                g_param = torch.zeros(param.shape)
                l_param = [torch.zeros(param.shape) for _ in range(args.world_size)]
            else:
                g_param = param.cpu()
                l_param = list()
                b_param = torch.zeros(param.shape)

            dist.gather(g_param, l_param, dst=args.master_rank)

            if args.is_master:
                b_param = sum(l_param) / (args.world_size - 1)
            dist.broadcast(b_param, src=args.master_rank)

            if not args.is_master:
                b_param = b_param.to(args.device)
            param *= 0
            param.set_(b_param)
    return model1, model2

import torch
from torch.autograd import grad

def gradient_penalty(critic, h_s, h_t,h_ref):
    alpha = torch.rand(h_s.size(0), 1,1).to(h_s.device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = interpolates.requires_grad_()
    h_refs = h_ref.requires_grad_()
    # interpolates = torch.cat([interpolates, h_s, h_t],dim=0).requires_grad_()
    # h_refs = torch.cat([h_ref,h_ref,h_ref],dim=0).requires_grad_()

    preds = critic(interpolates,h_refs)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty
"""Check the sampled PG estimator against the exact finite-action gradient."""
import torch
torch.manual_seed(20260922)
x=torch.tensor([.5,-.2,.1],requires_grad=True);reward=torch.tensor([1.,-2.,-.25])
prob=x.softmax(-1);exact=-(prob*reward).sum();exact.backward();target=x.grad.clone();x.grad=None
prob=x.softmax(-1);actions=torch.multinomial(prob.detach(),200000,replacement=True)
advantage=reward[actions]-(prob.detach()*reward).sum()
loss=-(advantage*prob.log()[actions]).mean();loss.backward()
error=(x.grad-target).abs().max().item()
assert error<.01,error
print({'max_gradient_error':error,'samples':len(actions),'status':'passed'})

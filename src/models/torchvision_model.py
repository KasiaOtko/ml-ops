import torch
import torchvision.models as models

model = models.resnet18(pretrained = True)
script_model = torch.jit.script(model)
script_model.save('models/deployable_model.pt')

input = torch.randn(1, 3, 224, 224)
unscripted_top5_indices = model(input).topk(5).indices
scripted_top5_indices = script_model(input).topk(5).indices
assert torch.allclose(unscripted_top5_indices, scripted_top5_indices)
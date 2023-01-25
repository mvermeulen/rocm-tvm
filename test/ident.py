# Basic example to generate identity as an ONNX graph
#
import torch
import torch.onnx

class SmallModel(torch.nn.Module):
    def __init__(self):
        super(SmallModel,self).__init__()
        self.ident1 = torch.nn.Identity()


    def forward(self, x):
        x = self.ident1(x)
        return x

smallmodel = SmallModel()

print('Model: ',smallmodel)
print('Params: ')
for param in smallmodel.parameters():
    print(param)

torch.onnx.export(smallmodel,
                  torch.randn(10),
                  'identity.onnx',
                  input_names = ['input'],
                  output_names = ['output'])
                  
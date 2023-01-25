# Basic example to generate identity as an ONNX graph
#
import torch
import torch.onnx

class SmallModel(torch.nn.Module):
    def __init__(self):
        super(SmallModel,self).__init__()
        self.blocks = torch.nn.Identity()


    def forward(self, x,y):
        x = x + y
        return x

smallmodel = SmallModel()

print('Model: ',smallmodel)
print('Params: ')
for param in smallmodel.parameters():
    print(param)

torch.onnx.export(smallmodel,
                  (torch.randn(10),torch.randn(10)),
                  'vector_add.onnx',
                  input_names = ['input1','input2'],
                  output_names = ['output'])
                  

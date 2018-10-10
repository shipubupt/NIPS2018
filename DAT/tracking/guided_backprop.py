import torch
from torch.nn import ReLU
from torch.autograd import Variable


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, target_class):
        self.model = model
        self.target_class = target_class
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer

        first_layer = list(self.model._modules.items())[0][1][0][1]
       # print first_layer
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in self.model._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self,input_image,vector):

        self.model.eval()
        # Forward pass
        model_output = self.model(input_image,out_layer='fc6')
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = Variable(torch.Tensor([vector]*input_image.size(0)))
        one_hot_output = one_hot_output.cuda()
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        return self.gradients


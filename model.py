import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class first_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, output_dim, class_num):
        super(first_Model, self).__init__()
        self.feature_extract_1 = nn.Linear(input_dim, hidden_dim1)
        self.feature_extract_2 = nn.Linear(hidden_dim1, hidden_dim2)

        self.regression_1 = nn.Linear(hidden_dim2, hidden_dim3)
        self.regression_2 = nn.Linear(hidden_dim3, hidden_dim4)
        self.regression_predict = nn.Linear(hidden_dim4, output_dim)

        self.domain_classifier_1 = nn.Linear(hidden_dim2, hidden_dim5)
        self.domain_predict = nn.Linear(hidden_dim5, class_num)
    
    def forward(self, x, alpha):
        feature = torch.tanh(self.feature_extract_1(x))
        feature = torch.tanh(self.feature_extract_2(feature))

        reverse_feature = ReverseLayerF.apply(feature, alpha)

        regression_out = torch.tanh(self.regression_1(feature))
        regression_out = torch.tanh(self.regression_2(regression_out))
        regression_out = self.regression_predict(regression_out)

        doamin_out = torch.tanh(self.domain_classifier_1(reverse_feature))
        doamin_out = torch.log(F.softmax(self.domain_predict(doamin_out), dim = 1))
        return regression_out, doamin_out

class ClassifierModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ClassifierModel, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim1)
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.predict = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = torch.tanh(self.hidden1(x))
        out = torch.tanh(self.hidden2(out))
        out = torch.log(F.softmax(self.predict(out), dim = 1))
        return out


class CcFTL(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, hidden_dim6, num_class):
        super(CcFTL, self).__init__()
        self.feature_extract_1 = nn.Linear(input_dim, hidden_dim1)
        self.feature_extract_2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.regression_1 = nn.Linear(hidden_dim2, hidden_dim3)
        self.regression_2 = nn.Linear(hidden_dim3, hidden_dim4)
        self.regression_predict = nn.Linear(hidden_dim4, 1)

        self.hidden1 = nn.Linear(input_dim + 1, hidden_dim5)
        self.hidden2 = nn.Linear(hidden_dim5, hidden_dim6)
        self.predict = nn.Linear(hidden_dim6, num_class)

    def forward(self, x):
        feature = torch.tanh(self.feature_extract_1(x))
        feature = torch.tanh(self.feature_extract_2(feature))

        regression_out = torch.tanh(self.regression_1(feature))
        regression_out = torch.tanh(self.regression_2(regression_out))
        regression_out = self.regression_predict(regression_out)

        inter = torch.cat((x, regression_out), 1)

        out = torch.tanh(self.hidden1(inter))
        out = torch.tanh(self.hidden2(out))
        out = torch.log(F.softmax(self.predict(out), dim=1))

        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

from .genotypes import *

# all node operations has two input and one output, 2C -> C
STEP_STEP_OPS = {
    'Sum': lambda C, L, args: Sum(),
    'ScaleDotAttn': lambda C, L, args: ScaledDotAttn(C, L),
    'LinearGLU': lambda C, L, args: LinearGLU(C, args),
    'ConcatFC': lambda C, L, args: ConcatFC(C, args),
    'SE1' : lambda C, L, args: SE1(C),
    'CatConvMish': lambda C, L, args: CatConvMish(C, args),
    'SE2' : lambda C, L, args: SE2(C),
    'LowRankTensorFusion': lambda C, L, args: LowRankTensorFusion(C, L,args)
}

class Sum(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        
        out = x + y
        
        return out

class LinearGLU(nn.Module):
    def __init__(self, C, args):
        super().__init__()
        # 1x1 conv1d
        self.conv = nn.Conv1d(2*C, 2*C, 1, 1)
        self.bn = nn.BatchNorm1d(2*C)
        self.dropout = nn.Dropout(args.drpt)

    def forward(self, x, y):
        
        # concat on channels
        out = torch.cat([x, y], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        
        # apply glu on channel dim
        out = F.glu(out, dim=1)
        out = self.dropout(out)
        
        return out

class ConcatFC(nn.Module):
    def __init__(self, C, args):
        super().__init__()
        # 1x1 conv1d
        self.conv = nn.Conv1d(2*C, C, 1, 1)
        self.bn = nn.BatchNorm1d(C)
        self.dropout = nn.Dropout(args.drpt)

    def forward(self, x, y):
        
        # concat on channels
        out = torch.cat([x, y], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = x * (torch.tanh(F.softplus(x)))
        return out 

class CatConvMish(nn.Module):
    def __init__(self, C, args):
        super().__init__()
        # 1x1 conv1d
        self.conv = nn.Conv1d(2*C, C, 1, 1)
        self.bn = nn.BatchNorm1d(C)
        self.dropout = nn.Dropout(args.drpt)
        self.mish = Mish()

    def forward(self, x, y):
        # concat on channels
        out = torch.cat([x, y], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = self.mish(out)
        out = self.dropout(out)
        return out

class ScaledDotAttn(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, C, L):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm([C, L])

    def forward(self, x, y):
        
        
        # trans pose C to last dim
        q = x.transpose(1, 2)
        k = y
        v = y.transpose(1, 2)
        
        d_k = q.size(-1)
        scores = torch.matmul(q, k) / math.sqrt(d_k)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2)
        out = self.dropout(out)
        out = self.ln(out)
        return out

class NodeMixedOp(nn.Module):
    def __init__(self, C, L, args):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in STEP_STEP_PRIMITIVES:
            op = STEP_STEP_OPS[primitive](C, L, args)
            self._ops.append(op)

    def forward(self, x, y, weights):
        out = sum(w * op(x, y) for w, op in zip(weights, self._ops))
        return out





############################################################################################################################
    
def init_weights(m):
 if type(m) != nn.Linear:
   print('error')


# This multimodal fusion operator will starts by squeezing the x tensor and then use it to rescaling (exite) the y tensor 
class SE1(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.fc_squeeze = nn.Linear(C, C)
        
        with torch.no_grad():
            self.fc_squeeze.apply(init_weights)    
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        
    def forward(self, x, y):
        
        tview = x.view(x.shape[:2] + (-1,))
        squeeze = torch.mean(tview, dim=-1)
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)
        excitation = self.sigmoid(excitation)

        dim_diff = len(y.shape) - len(excitation.shape)
        excitation = excitation.view(excitation.shape + (1,) * dim_diff)
        out = y * excitation
        
        return out


# This multimodal fusion operator will starts by squeezing the y tensor and then use it to rescaling (exite) the x tensor 
class SE2(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.fc_squeeze = nn.Linear(C, C)
        
        with torch.no_grad():
            self.fc_squeeze.apply(init_weights)    
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        
    def forward(self, y, x):
        
        tview = x.view(x.shape[:2] + (-1,))
        squeeze = torch.mean(tview, dim=-1)
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)
        excitation = self.sigmoid(excitation)

        dim_diff = len(y.shape) - len(excitation.shape)
        excitation = excitation.view(excitation.shape + (1,) * dim_diff)
        out = y * excitation
        
        return out


############################################################################################################################
    
class LowRankTensorFusion(nn.Module):
    """
    Implementation of Low-Rank Tensor Fusion.
    
    See https://github.com/Justin1904/Low-rank-Multimodal-Fusion for more information.
    """

    def __init__(self, C, L, args, flatten=True):
        """
        Initialize LowRankTensorFusion object.
        
        :param input_dims: list or tuple of integers indicating input dimensions of the modalities
        :param output_dim: output dimension
        :param rank: a hyperparameter of LRTF. See link above for details
        :param flatten: Boolean to dictate if output should be flattened or not. Default: True
        
        """
        super(LowRankTensorFusion, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.C = C
        self.L = L
        self.input_dims = [C*L, C*L]
        self.output_dim = C*L
        self.rank = 16
        self.flatten = flatten
        self.args = args

        # low-rank factors
        self.factors = []
        for input_dim in self.input_dims:
            factor = nn.Parameter(torch.Tensor(
                # self.rank, input_dim+1, self.output_dim)).to(device)
                self.rank, input_dim+1, self.output_dim))
            nn.init.xavier_normal_(factor)
            self.factors.append(factor)

        # self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank)).to(device)
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(
            # torch.Tensor(1, self.output_dim)).to(device)
            torch.Tensor(1, self.output_dim))
        # init the fusion weights
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, x, y):
        """
        Forward Pass of Low-Rank TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        modalities = [x.view(-1, self.C*self.L), y.view(-1, self.C*self.L)]
        batch_size = modalities[0].shape[0]
        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        fused_tensor = 1
        for (modality, factor) in zip(modalities, self.factors):
            ones = Variable(torch.ones(batch_size, 1).type(
                # modality.dtype), requires_grad=False).to(modality.device)
                modality.dtype), requires_grad=False)
            if self.flatten:
                modality_withones = torch.cat(
                    (ones.to(torch.device("cuda:"+str(self.args.gpu))), torch.flatten(modality.to(torch.device("cuda:"+str(self.args.gpu))), start_dim=1))  , dim=1)
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones.to(torch.device("cuda:"+str(self.args.gpu))), factor.to(torch.device("cuda:"+str(self.args.gpu))))
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(
            1, 0, 2)).squeeze() + self.fusion_bias
        #output = output.view(-1, self.output_dim)
        output = output.view(-1, self.C, self.L)
        return output
    
if __name__ == '__main__':
    model = LowRankTensorFusion(C=192, L=16)
    _input = torch.rand(8, 192, 16)
    print(model(_input, _input).size())
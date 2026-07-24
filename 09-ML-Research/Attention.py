import torch 
import math
import torch.nn.functional as F

def dotProduct_Attention(Q, K, V ):
    dk = K.shape[-1]
    scores = torch.matmul(Q,K.transpose(-2, -1))/math.sqrt(dk)
    weights = F.softmax(scores , dim=-1)
    output = torch.matmul(weights,V)

    return weights , output

Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
weights , output = dotProduct_Attention(Q, K, V)    

print("Attention weights shape:", weights.shape)
print("Output shape:", output.shape)
print("Attention weights (first head):\n", weights[0])
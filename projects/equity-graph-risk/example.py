import torch
from risk_model import NodeTransformerEncoder, GNNModel

if __name__ == "__main__":
    torch.manual_seed(7)
    history = torch.randn(1, 5, 36, 12)
    edges = torch.eye(5).unsqueeze(0)
    encoder = NodeTransformerEncoder().eval()
    model = GNNModel().eval()
    with torch.no_grad():
        correlations = model(encoder(history), edges)[0]
    assert correlations.shape == (5, 5)
    assert torch.allclose(correlations, correlations.T)
    assert torch.allclose(correlations.diag(), torch.ones(5))
    print(correlations)

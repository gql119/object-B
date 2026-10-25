"""Explicit COCO-only selected-coordinate version; projection mathematics unchanged."""
import torch

PROTOCOL = 'coco80-strict-tsfl-selected-dimension-v1'
RESULT_LABEL = 'Original S4 — COCO80 Dimension-Compatible Implementation'


def selected_dimension(indices, feature_channels, *, rank=8):
    if (not torch.is_tensor(indices) or indices.ndim != 1 or indices.dtype != torch.long
            or indices.numel() < rank or indices.numel() > feature_channels):
        raise ValueError('Selected coordinates cannot support the fixed Safe-BG rank')
    if (int(indices.min()) < 0 or int(indices.max()) >= feature_channels
            or not torch.equal(indices, indices.sort().values)
            or indices.unique().numel() != indices.numel()):
        raise ValueError('Selected coordinates must be unique, ordered and in the bound P5 space')
    return int(indices.numel())


def validate_identity(identity, indices, feature_channels):
    dimension = selected_dimension(indices, feature_channels)
    expected = dict(dimension_protocol=PROTOCOL, selected_space_dimension=dimension,
                    selected_channel_indices=indices.cpu().tolist(), num_classes=80, target_class_id=0)
    if any(identity.get(k) != value for k, value in expected.items()):
        raise ValueError('COCO selected-space artifact protocol/coordinate identity mismatch')
    return dimension


def validate_receipt_dimension(receipt):
    indices = torch.tensor(receipt.get('selected_channels', []), dtype=torch.long)
    dimension = selected_dimension(indices, receipt.get('total_feature_channels', 0))
    if (receipt.get('dimension_protocol') != PROTOCOL
            or receipt.get('selected_channel_count') != dimension):
        raise ValueError('COCO selected-dimension receipt identity mismatch')
    return dimension

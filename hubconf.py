import torch
from models.backbone import Backbone, Joiner
from models.deformable_detr import DeformableDETR, PostProcess
from models.position_encoding import PositionEmbeddingSine
from models.deformable_transformer import DeformableTransformer

dependencies = ["torch", "torchvision"]


def _make_deformable_detr(backbone_name: str, dilation=False, num_classes=91, object_embedding_loss=False):
    hidden_dim = 256
    backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=True, dilation=dilation,
                        load_backbone='swav')
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = DeformableTransformer(d_model=hidden_dim, return_intermediate_dec=True)
    deformable_detr = DeformableDETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=300,
                                     num_feature_levels=4, object_embedding_loss=object_embedding_loss,
                                     obj_embedding_head='intermediate')
    return deformable_detr


def detreg_resnet50(pretrained=False, return_postprocessor=False, num_classes=91, object_embedding_loss=False,
                             checkpoints_path=""):
    """
    Deformable DETR R50 with 6 encoder and 6 decoder layers.
    """
    model = _make_deformable_detr("resnet50", dilation=False, num_classes=num_classes,
                                  object_embedding_loss=object_embedding_loss)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    elif checkpoints_path:
        checkpoint = torch.load(checkpoints_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model

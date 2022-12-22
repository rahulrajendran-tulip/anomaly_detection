# type: ignore
# import timm  # pylint: disable=arguments-differ
import torch

print(torch.__version__)


# # Model Weights test
# path_to_model = (
#     "./results/lightning_logs/version_0/checkpoints/epoch001-val_f1 0.9133.ckpt"
# )
# model = timm.create_model(
#     model_name="resnet18",
#     pretrained=True,
#     num_classes=2,
# )

# # for values in model.state_dict():
# #     print(values, "\t", model.state_dict()[values].size())
# # # model = torchvision.models.resnet18(pretrained=True)

# # for param in model.parameters():
# #     print(param.data)

# # model.load_state_dict(torch.load(path_to_model))
# # for param in model.parameters():
# #     print(param.data)


# state_dict = torch.load(path_to_model)
# prefix = "model."
# n_clip = len(prefix)
# adapted_dict = {
#     k[n_clip:]: v for k, v in state_dict["state_dict"].items() if k.startswith(prefix)
# }

# model.load_state_dict(adapted_dict)
# for param in model.parameters():
#     print(param.data)
# # for k, v in state_dict["state_dict"].items():
# #     if k.startswith(prefix):
# #         adapted_dict = {(v, k[n_clip:])}
# # model = torch.load(path_to_model)
# print(model)

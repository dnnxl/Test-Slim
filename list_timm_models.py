import timm

# list models
# m = timm.list_models('*swin*',pretrained=True)
# m = m[0:500]
# for i in m:
#     print(f"Model: {i}")


# m = [
#         "resnet10t.c3_in1k",
#         "resnet18",
#         "resnetv2_50", 
#         "swinv2_base_window8_256.ms_in1k",
#         "resnetrs420",
#         "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k"
#     ]
m = ['resnetrs420']
for i in m:
    model = timm.create_model(i, pretrained=False).cuda()
    num = sum(p.numel() for p in model.parameters())
    print(f"Model: {i}. # params: {num}")
    print()

    
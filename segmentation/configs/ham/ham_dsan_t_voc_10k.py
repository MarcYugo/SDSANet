_base_ = [
    '../_base_/models/sdsan_ham.py',
    '../_base_/datasets/voc.py',
    '../_base_/default_runtime.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/sdsan_t.pth')),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        ham_kwargs=dict(MD_R=16),
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(type='AdamW', lr=0.00012, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=1e-6, by_epoch=False)
# fp16
fp16 = dict(loss_scale='dynamic')
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU')
data = dict(samples_per_gpu=8)

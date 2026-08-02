class DictToObj:
    def __init__(self, d):
        self.__dict__.update(d)

FLAGS = DictToObj({
    "num_gpus_per_job": 2,
    "num_cpus_per_job": 63,
    "memory_per_job": 380,
    "gpu_type": "nvidia-tesla-t4",
    "dataset": "cifar10",
    "num_classes": 10,
    "image_size": 32,
    "topk": [1, 5],
    "num_epochs": 100,
    "optimizer": "sgd",
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "nesterov": True,
    "lr": 0.1,
    "lr_scheduler": "multistep",
    "multistep_lr_gamma": 0.1,
    "profiling": ["gpu"],
    "pretrained": "logs/us_resnet50_0.25_1.0.pt",
    "resume": "",
    "test_only": False,
    "random_seed": 1995,
    "batch_size": 256,
    "model_name": "us_resnet",
    "reset_parameters": True,
    "log_dir": "logs/",
    "slimmable_training": True,
    "depth": 50,

    # --- universally slimmable settings ---
    "width_mult_range": (0.25, 1.0),      # continuous training range [min, max]
    "num_sample_widths": 2,               # extra random (width, bit_width) pairs per sandwich step
    "width_divisor": 8,                   # round channel counts to a multiple of this

    # --- quantization-aware training settings ---
    "bit_width_list": [4, 8, 16, 32],     # 32 == full precision, always used for the teacher
    "deploy_configs": [
        (0.25, 4), (0.25, 8), (0.25, 16), (0.25, 32),
        (0.5, 4),  (0.5, 8),  (0.5, 16),  (0.5, 32),
        (0.75, 4), (0.75, 8), (0.75, 16), (0.75, 32),
        (1.0, 4),  (1.0, 8),  (1.0, 16),  (1.0, 32),
    ],
    "bn_calibration_widths": [0.25, 0.5, 0.75, 1.0],
    "bn_calibration_bits": [4, 8, 16, 32]
})

FLAGS.multistep_lr_milestones = [int(FLAGS.num_epochs * 0.5), int(FLAGS.num_epochs * 0.75)]

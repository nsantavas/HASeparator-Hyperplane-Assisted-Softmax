parameters = {
    "learning_rate": 0.1,
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 1E-4,
    "input_shape": (32, 32, 3),
    "dropout": 0.3,
    "batch_size": 128,
    "drop_remainder": True,
    "shuffle_size": 1000,
    "total_classes": 10,
    "margin": 0.1,
    "scaler": 1,
    "logdir": './train_logs',
    "filepath": './models/weights.hdf5',
    "epochs": 150
}
import tensorflow as tf
from params import parameters


class Dataloader():
    def __init__(self):
        (self.x_train,
         self.y_train), (self.x_test,
                         self.y_test) = tf.keras.datasets.cifar10.load_data()

        global AUTO
        AUTO = tf.data.experimental.AUTOTUNE

    @staticmethod
    def load_dataset(x, y):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.map(map_func=Dataloader.augmentation,
                              num_parallel_calls=AUTO)
        return dataset

    @staticmethod
    def augmentation(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        return (image, label), label

    @staticmethod
    def get_batched_dataset(x, y):
        dataset = Dataloader.load_dataset(x, y)
        dataset = dataset.shuffle(parameters["shuffle_size"])
        dataset = dataset.batch(parameters["batch_size"],
                                drop_remainder=parameters['drop_remainder'])
        dataset = dataset.prefetch(AUTO)
        return dataset

    def get_training_dataset(self):
        return Dataloader.get_batched_dataset(self.x_train, self.y_train)

    def get_validation_dataset(self):
        return Dataloader.get_batched_dataset(self.x_test, self.y_test)
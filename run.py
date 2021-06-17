import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from ResNet50 import ResNet50, scheduler
from Dataloader import Dataloader
from params import parameters

dataloader = Dataloader()  # Initializing Dataloader

scheduler = tf.keras.callbacks.LearningRateScheduler(
    scheduler)  # Initializing scheduler

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=parameters['logdir'],
    write_graph=True)  # Initializing TensorBoard Callback

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    (parameters['filepath']),
    mode='max',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    save_weights_only=True)  # Initializing checkpoint Callback

clbk = [tensorboard_callback, checkpoint, scheduler]  # List of Callbacks

resnet = ResNet50(parameters["total_classes"], parameters["margin"],
                  parameters["scaler"]) # Initializing ResNet50 class

model = resnet.model() # # Initializing ResNet50 model

if __name__ == "__main__":

    model.fit(dataloader.get_training_dataset(),
              validation_data=dataloader.get_validation_dataset(),
              epochs=parameters["epochs"],
              verbose=1,
              callbacks=clbk)

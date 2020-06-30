import time

import numpy
import tensorflow
from matplotlib import pyplot
from tensorflow.keras import layers

from simple import generator_loss, discriminator_loss

BUFFER_SIZE = 20000
BATCH_SIZE = 256
EPOCHS = 10001
noise_dim = 100
num_examples_to_generate = 10000

seed = tensorflow.random.normal([num_examples_to_generate, noise_dim])


class QuadraticGAN:

    def __init__(self) -> None:
        super().__init__()
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)

    def train_step(self, x_batch):
        noise = tensorflow.random.normal([BATCH_SIZE, noise_dim])

        with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)
            x = x_batch  # .reshape(x_batch.shape[0], 2, 1, 1)
            real_output = self.discriminator(x, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, dataset, epochs):
        # train_images = self.train_images.reshape(self.train_images.shape[0], 28, 28, 1).astype('float32')
        for epoch in range(epochs):
            start = time.time()

            for x_batch in dataset:
                self.train_step(x_batch)
            # print(epoch)
            if epoch % 5 == 0:
                print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                # f.write("%d,%f,%f\n" % (i, dloss, gloss))

            if epoch % 20 == 0:
                # Produce images for the GIF as we go
                # display.clear_output(wait=True)
                self.generate_and_save_images(self.generator, epoch + 1, seed)


        # Generate after the final epoch
        # display.clear_output(wait=True)
        self.generate_and_save_images(self.generator, epochs, seed)


    def run(self):
        y = self.sample_data()
        train_dataset = tensorflow.data.Dataset.from_tensor_slices(y).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        self.train(train_dataset, EPOCHS)

    @staticmethod
    def get_y(x):
        return x * x + 10

    @staticmethod
    def sample_data(n=num_examples_to_generate, scale=100):
        data = []
        x = (numpy.random.random((n,)) - .5) * scale
        for i in range(n):
            data.append((x[i], QuadraticGAN.get_y(x[i])))
        return numpy.array(data)

    @staticmethod
    def make_generator_model():
        model = tensorflow.keras.Sequential()
        model.add(layers.Dense(16, use_bias=False, input_shape=(100,)))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(16, use_bias=False))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(2, use_bias=False))
        return model

    @staticmethod
    def make_discriminator_model():
        model = tensorflow.keras.Sequential()
        model.add(layers.Dense(16, use_bias=False, input_shape=(2,)))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(16, use_bias=False))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(2, use_bias=False))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(2, use_bias=False))
        model.add(layers.Dense(1, use_bias=False))
        return model

    @staticmethod
    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        pyplot.scatter(*zip(*predictions.numpy()))
        # pyplot.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        pyplot.show()

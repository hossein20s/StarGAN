import time

import numpy
import tensorflow
from matplotlib import pyplot
from tensorflow.keras import layers

from simple import generator_loss, discriminator_loss

BUFFER_SIZE = 200
BATCH_SIZE = 10
EPOCHS = 10001
noise_dim = 100
num_examples_to_generate = 100

seed = tensorflow.random.normal([num_examples_to_generate, noise_dim])


class QuadraticGAN:

    def __init__(self) -> None:
        super().__init__()
        self.data = self.sample_data()
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)

    def train_step(self, x):
        noise = tensorflow.random.normal([BATCH_SIZE, noise_dim])

        with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)
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
        start = time.time()
        for epoch in range(epochs):

            for x_batch in dataset:
                self.train_step(x_batch)
            # print(epoch)
            if epoch % 20 == 0:
                print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                # f.write("%d,%f,%f\n" % (i, dloss, gloss))

            if epoch % 100 == 0:
                # Produce images for the GIF as we go
                # display.clear_output(wait=True)
                self.log_loss_and_save_images(epoch + 1, seed)

        # Generate after the final epoch
        # display.clear_output(wait=True)
        self.log_loss_and_save_images(epochs, seed)

    def run(self):
        train_dataset = tensorflow.data.Dataset.from_tensor_slices(self.data).batch(BATCH_SIZE)
        self.train(train_dataset, EPOCHS)

    @staticmethod
    def get_y(x):
        return x * x + 0

    @staticmethod
    def sample_data(n=num_examples_to_generate, scale=10):
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

    def log_loss_and_save_images(self, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        generated_data = self.generator(test_input, training=False)
        real_output = self.discriminator(self.data, training=False)
        fake_output = self.discriminator(generated_data, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        print('epoch {}, gen loss {}, discriminator loss {}'.format(epoch, gen_loss.numpy(), disc_loss.numpy()))

        # pyplot.figure(figsize=(1, 2))

        pyplot.scatter(*zip(*generated_data.numpy()))
        pyplot.scatter(*zip(*self.data))
        pyplot.show()
        # pyplot.plot(*zip(*fake_output.numpy()))
        # pyplot.plot(*zip(*real_output.numpy()))
        # # pyplot.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        # pyplot.show()

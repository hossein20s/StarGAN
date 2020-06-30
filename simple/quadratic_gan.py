import numpy
import tensorflow
from tensorflow.keras import layers

from simple import generator_loss, discriminator_loss

BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100


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
            print(generated_data.shape)
            real_output = self.discriminator(x, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, epochs):
        # train_images = self.train_images.reshape(self.train_images.shape[0], 28, 28, 1).astype('float32')
        for epoch in range(epochs):
            self.train_step(self.sample_data(n=BATCH_SIZE))

    def run(self):
        self.train(EPOCHS)

    @staticmethod
    def get_y(x):
        return x * x + 10

    @staticmethod
    def sample_data(n=1000, scale=100):
        data = []
        x = (numpy.random.random((n,)) - .5) * scale
        for i in range(n):
            data.append((x[i], QuadraticGAN.get_y(x[i])))
        return numpy.array(data)

    @staticmethod
    def make_generator_model():
        model = tensorflow.keras.Sequential()
        model.add(layers.Dense(16, use_bias=False, input_shape=(BATCH_SIZE, 100)))
        model.add(layers.LeakyReLU())
        # model.add(layers.Dense(16, use_bias=False))
        # model.add(layers.LeakyReLU())
        model.add(layers.Dense(2, use_bias=False))
        return model

    @staticmethod
    def make_discriminator_model():
        model = tensorflow.keras.Sequential()
        model.add(layers.Dense(16, use_bias=False, input_shape=(BATCH_SIZE, 2)))
        model.add(layers.LeakyReLU())
        # model.add(layers.Dense(16, use_bias=False))
        # model.add(layers.LeakyReLU())
        # model.add(layers.Dense(2, use_bias=False))
        # model.add(layers.LeakyReLU())
        model.add(layers.Dense(2, use_bias=False))
        model.add(layers.Dense(1, use_bias=False))
        return model

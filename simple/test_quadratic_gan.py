from unittest import TestCase

import tensorflow
from matplotlib import pyplot

from simple.quadratic_gan import QuadraticGAN


class TestQuadraticGAN(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.gan = QuadraticGAN()

    def test_sample_data(self):
        y = self.gan.sample_data()
        print(y.shape)
        pyplot.scatter(*zip(*y))
        pyplot.show()

    def test_make_generator_model(self):
        generator = self.gan.make_generator_model()

        noise = tensorflow.random.normal([1000, 100])
        y = generator(noise, training=False)
        print(y.shape)

        pyplot.scatter(*zip(*y.numpy()))
        pyplot.show()

    def test_make_discriminator_model(self):
        generator = self.gan.make_generator_model()

        noise = tensorflow.random.normal([1000, 100])
        y = generator(noise, training=False)
        discriminator = self.gan.make_discriminator_model()
        decision = discriminator(y)
        print(decision)

    def test_run(self):
        self.gan.run(debug=False)



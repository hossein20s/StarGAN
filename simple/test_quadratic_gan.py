from unittest import TestCase

import tensorflow
from matplotlib import pyplot

from simple.quadratic_gan import QuadraticGAN


class TestQuadraticGAN(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.gan = QuadraticGAN()

    def test_sample_data(self):
        data = self.gan.sample_data()
        pyplot.scatter(*zip(*data))
        pyplot.show()

    def test_make_generator_model(self):
        generator = self.gan.make_generator_model()

        noise = tensorflow.random.normal([1, 100])
        generated_image = generator(noise, training=False)

        pyplot.imshow(generated_image[0, :, :, 0], cmap='gray')
        pyplot.show()

    def test_make_discriminator_model(self):
        generator = self.gan.make_generator_model()

        noise = tensorflow.random.normal([1, 100])
        generated_image = generator(noise, training=False)
        discriminator = self.gan.make_discriminator_model()
        decision = discriminator(generated_image)
        print(decision)

    def test_run(self):
        self.gan.run()

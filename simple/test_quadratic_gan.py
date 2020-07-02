from unittest import TestCase

import tensorflow
from matplotlib import pyplot

from simple.quadratic_gan import QuadraticGAN


def get_y(x):
    return x * x + 0


def is_part_of_line(x):
    initial = (1, 2)
    final = (8, 10)
    if x < initial[0] or x > final[0]:
        return 0
    m = (final[1] - initial[1]) / (final[0] - initial[0])
    b = initial[1] - m * initial[0]
    return m * x + b


class TestQuadraticGAN(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.gan = QuadraticGAN(target_function=is_part_of_line, initial_batch_size=3, number_of_batch=10, epochs=100,
                                number_of_interation=3)

    def test_sample_data(self):
        y = self.gan.sample_data(number_of_sample=100, scale=10)
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

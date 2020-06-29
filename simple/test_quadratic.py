from unittest import TestCase

import tensorflow
from matplotlib import pyplot

from simple.quadratic import sample_data, make_generator_model, make_discriminator_model, GAN, run_simple, SimpleGAN


class Test(TestCase):
    def test_sample_data(self):
        data = sample_data()
        pyplot.scatter(*zip(*data))
        pyplot.show()

    def test_make_generator_model(self):
        generator = make_generator_model()

        noise = tensorflow.random.normal([1, 100])
        generated_image = generator(noise, training=False)

        pyplot.imshow(generated_image[0, :, :, 0], cmap='gray')
        pyplot.show()

    def test_make_discriminator_model(self):
        generator = make_generator_model()

        noise = tensorflow.random.normal([1, 100])
        generated_image = generator(noise, training=False)
        discriminator = make_discriminator_model()
        decision = discriminator(generated_image)
        print(decision)

    def test_run_simple(self):
        run_simple()


class TestGAN(TestCase):
    def test_run(self):
        gan = GAN()
        pyplot.imshow(gan.train_images[0])
        pyplot.show()
        gan.run()


class TestSimpleGAN(TestCase):
    def test_run(self):
        gan = SimpleGAN()
        gan.run()

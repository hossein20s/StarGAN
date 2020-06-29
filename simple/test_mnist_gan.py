from unittest import TestCase

from simple.quadratic_gan import QuadraticGAN


class TestMnistGAN(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.gan = QuadraticGAN()

    def test_run(self):
        self.gan.run()

import io
import logging
import time

import numpy
import tensorflow
from matplotlib import animation
from matplotlib import pyplot
from tensorflow.keras import layers

from simple import generator_loss, discriminator_loss

BUFFER_SIZE = 36
BATCH_SIZE = 9
EPOCHS = 20000
noise_dim = 100
num_examples_to_generate = 36

seed = tensorflow.random.normal([num_examples_to_generate, noise_dim])


class QuadraticGAN:

    def __init__(self) -> None:
        super().__init__()
        logging.basicConfig(filename='/opt/host/Downloads/quadtaricGAN.log', level=logging.DEBUG)
        self.logger = logging.getLogger(self.__class__.__name__)
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

    def train(self, dataset, epochs, debug=False):
        start = time.time()
        image_buffers = []
        for epoch in range(epochs):

            for x_batch in dataset:
                self.train_step(x_batch)
            # self.logger.info(epoch)
            if epoch % 20 == 0:
                self.logger.info('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                # f.write("%d,%f,%f\n" % (i, dloss, gloss))

            if epoch % 100 == 0:
                # Produce images for the GIF as we go
                # display.clear_output(wait=True)
                image_buffers.append(self.log_loss_and_save_images(epoch + 1, seed, debug=debug))
        # Generate after the final epoch
        # display.clear_output(wait=True)
        image_buffers.append(self.log_loss_and_save_images(epochs, seed, debug=debug))
        return image_buffers

    def run(self, debug=False):
        train_dataset = tensorflow.data.Dataset.from_tensor_slices(self.data).batch(BATCH_SIZE)
        image_buffers = self.train(train_dataset, EPOCHS, debug=debug)
        # First set up the figure, the axis, and the plot element we want to animate
        fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(10, 10))
        pyplot.close()
        ax.xlim = (0, 10)
        ax.ylim = (0, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        buffer = image_buffers[0]
        buffer.seek(0)
        img = ax.imshow(pyplot.imread(buffer))
        img.set_interpolation('nearest')

        def updatefig(frame, buffers):
            self.logger.info(frame)
            buffer = buffers[frame]
            buffer.seek(0)
            img.set_data(pyplot.imread(buffer))
            return (img,)

        animation_function = animation.FuncAnimation(fig, updatefig, frames=len(image_buffers), fargs=(image_buffers,),
                                                     interval=100, blit=True)
        anim_writer = animation.writers['ffmpeg']
        writer = anim_writer(fps=15, metadata=dict(artist='Hossein'), bitrate=1800)
        time_string = time.strftime("%Y%m%d-%H%M%S")
        file_name = '{}{}-{}.mp4'.format('/opt/host/Downloads/', QuadraticGAN.__name__, time_string)
        animation_function.save(file_name, writer=writer, )

    def log_loss_and_save_images(self, epoch, test_input, debug=False):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        generated_data = self.generator(test_input, training=False)
        real_output = self.discriminator(self.data, training=False)
        fake_output = self.discriminator(generated_data, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        self.logger.info(
            'epoch {}, gen loss {}, discriminator loss {}'.format(epoch, gen_loss.numpy(), disc_loss.numpy()))
        pyplot.close()
        pyplot.scatter(*zip(*generated_data.numpy()))
        pyplot.scatter(*zip(*self.data))
        buffer = io.BytesIO()
        pyplot.savefig(buffer, format='png')
        if debug:
            buffer.seek(0)
            pyplot.imread(buffer)
            pyplot.show()
        return buffer

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

    # def simple_animate(self):
    #     arr = []
    #     for i in range(100):
    #         c = numpy.random.rand(10, 10)
    #         arr.append(c)
    #     fig = pyplot.figure()
    #     im = pyplot.imshow(arr[0], animated=True)
    #
    #     def updatefig(*args):
    #         global i
    #         if (i < 99):
    #             i += 1
    #         else:
    #             i = 0
    #         im.set_array(arr[i])
    #         return im,
    #
    #     ani = animation.FuncAnimation(fig, updatefig, blit=True)
    #     pyplot.show()

    # def updatefig(*args):
    #     pyplot.scatter(*zip(*generated_data.numpy()))
    #     pyplot.scatter(*zip(*self.data))
    #     buf = io.BytesIO()
    #     pyplot.savefig(buf, format='png')
    #     buf.seek(0)
    #     return pyplot.imshow(pyplot.imread(buf)),

    # # pyplot.figure(figsize=(1, 2))
    # # pyplot.show()
    #
    # animation_function = animation.FuncAnimation(self.fig, updatefig, frames=100, repeat=True)
    # # pyplot.plot(*zip(*fake_output.numpy()))
    # # pyplot.plot(*zip(*real_output.numpy()))
    # # # pyplot.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # # updatefig()
    # # pyplot.show()
    # if last:
    #     animation_function.save('1.mp4', writer=self.anim_writer)

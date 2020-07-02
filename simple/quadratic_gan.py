import io
import logging
import logging.config
import os
import time

import numpy
import tensorflow
from matplotlib import animation
from matplotlib import pyplot
from tensorflow.keras import layers

from simple import generator_loss, discriminator_loss

SCALE = 10
noise_dim = 100
checkpoint_dir = '/opt/host/Downloads/training_checkpoints'

class QuadraticGAN:

    def __init__(self, initial_batch_size, number_of_batch) -> None:
        super().__init__()
        logging.config.fileConfig(fname='log.conf')
        self.logger = logging.getLogger('dev')
        self.batch_size = initial_batch_size
        self.number_of_batch = number_of_batch
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tensorflow.keras.optimizers.Adam(1e-4)
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tensorflow.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                                      discriminator_optimizer=self.discriminator_optimizer,
                                                      generator=self.generator,
                                                      discriminator=self.discriminator)

    def run(self, debug=False):
        self.run_iteration(batch_size=self.batch_size, epochs=1000, debug=debug)
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        # self.checkpoint.restore(tensorflow.train.latest_checkpoint(checkpoint_dir))
        self.run_iteration(batch_size=self.batch_size * 5, epochs=1000, debug=debug)
        self.run_iteration(batch_size=self.batch_size * 20, epochs=1000, debug=debug)
        self.run_iteration(batch_size=self.batch_size * 30, epochs=1000, debug=debug)

    def run_iteration(self, batch_size, epochs, debug=False):
        data = self.sample_data(number_of_sample=self.number_of_batch * batch_size, scale=SCALE)
        image_buffers = self.train(data=data, epochs=epochs, debug=debug)
        self.log_result(data=data, image_buffers=image_buffers)

    def train_step(self, x_batch):
        noise = tensorflow.random.normal([len(x_batch), noise_dim])

        with tensorflow.GradientTape() as gen_tape, tensorflow.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)
            real_output = self.discriminator(x_batch, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, data, epochs, debug=False):
        dataset = tensorflow.data.Dataset.from_tensor_slices(data).batch(self.batch_size)
        test_input = tensorflow.random.normal([len(data), noise_dim])
        start = time.time()
        image_buffers = []
        for epoch in range(epochs):

            for x_batch in dataset:
                self.train_step(x_batch)
            # self.logger.info(epoch)
            if epoch % max(20, epochs//10) == 0:
                self.logger.info('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                # f.write("%d,%f,%f\n" % (i, dloss, gloss))

            if epoch % min(100, epochs//10) == 0:
                # Produce images for the GIF as we go
                # display.clear_output(wait=True)
                image_buffers.append(
                    self.log_loss_and_save_images(epoch=epoch + 1, data=data, test_input=test_input, debug=debug))
        # Generate after the final epoch
        # display.clear_output(wait=True)
        image_buffers.append(self.log_loss_and_save_images(data=data, epoch=epochs, test_input=test_input, debug=debug))
        return image_buffers

    def log_result(self, data, image_buffers):
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

    def log_loss_and_save_images(self, data, epoch, test_input, debug=False):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        generated_data = self.generator(test_input, training=False)
        real_output = self.discriminator(data, training=False)
        fake_output = self.discriminator(generated_data, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        self.logger.info(
            'epoch {}, gen loss {}, discriminator loss {}'.format(epoch, gen_loss.numpy(), disc_loss.numpy()))
        pyplot.close()
        pyplot.scatter(*zip(*generated_data.numpy()))
        pyplot.scatter(*zip(*data))
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
    def sample_data(number_of_sample, scale):
        data = []
        x = (numpy.random.random((number_of_sample,)) - .5) * scale
        for i in range(number_of_sample):
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
    #     pyplot.scatter(*zip(*data))
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

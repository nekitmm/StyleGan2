import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from modconv import ModConv2D

def Discriminator(filters, image_size):
    multi = [1, 2, 4, 6, 8, 16, 32]
    def block(x, filters, pool = True):
        t = L.Conv2D(filters = filters, kernel_size = 1, kernel_initializer = 'he_uniform')(x)
        x = L.Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(x)
        x = L.LeakyReLU(0.2)(x)
        x = L.Conv2D(filters = filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(x)
        x = L.LeakyReLU(0.2)(x)
        x = L.add([x, t])
        
        if pool: x = L.AveragePooling2D()(x)
        
        return x
    
    inp = L.Input(shape = [image_size, image_size, 3])
    
    x = inp
    for i in range(6):
        x = block(x, filters * multi[i])
    
    x = block(x, filters * multi[-1], pool = False)
    x = L.Flatten()(x)
    x = L.Dense(1, kernel_initializer = 'he_uniform')(x)
    return K.Model(inputs = inp, outputs = x)
    
    
def G_map(ndims, nlayers):
    inp = L.Input(shape = (ndims,), name = 'mappping_input', dtype = tf.float32)
    l = inp
    for i in range(nlayers):
        l = L.Dense(ndims, name = 'mapping_dense_%d' % i, activation = tf.nn.leaky_relu)(l)
    return K.Model(inputs = inp, outputs = l, name = 'G_map')
    
def G_synthesis(image_size, nfilters, nlayers):
    def torgb(inputs, style):
        upsample = image_size // inputs.shape[2]
        x = ModConv2D(filters = 3, kernel_size = 1, demodulate = False)([inputs, style])
        x = L.UpSampling2D(size = (upsample, upsample), data_format = 'channels_last', interpolation = 'bilinear')(x)
        return x
    
    def block(inputs, style, noise, nfilters, demodulate = True, up = True):
        x = inputs
        
        if up: x = L.UpSampling2D(size=(2, 2), data_format = 'channels_last', interpolation = 'bilinear')(x)
        noise = L.Lambda(lambda x: x[0][:, :x[1].shape[1], :x[1].shape[2], :])([noise, x])
        
        s = L.Dense(inputs.shape[3], kernel_initializer = 'he_uniform')(style)
        d = L.Dense(nfilters, kernel_initializer = 'zeros')(noise)
        x = ModConv2D(filters = nfilters, kernel_size = 3, demodulate = demodulate)([x, s])
        
        x = L.add([x, d])
        x = L.LeakyReLU(alpha = 0.2)(x)
        
        s = L.Dense(nfilters, kernel_initializer = 'he_uniform')(style)
        d = L.Dense(nfilters, kernel_initializer = 'zeros')(noise)
        
        x = ModConv2D(filters = nfilters, kernel_size = 3, demodulate = demodulate)([x, s])
        x = L.add([x, d])
        x = L.LeakyReLU(alpha = 0.2)(x)
        
        s = L.Dense(nfilters, kernel_initializer = 'he_uniform')(style)
        
        return x, torgb(x, s)
    
    style_input = []
    for i in range(nlayers):
        style_input.append(K.Input([512]))
    
    noise_input = L.Input([image_size, image_size, 1])
    
    rgbs = []
    
    x = L.Lambda(lambda x: x[:, :1] * 0 + 1)(style_input[0])
    x = L.Dense(4 * 4 * 4 * nfilters, activation = 'relu', kernel_initializer = 'random_normal')(x)
    x = L.Reshape([4, 4, 4 * nfilters])(x)
    
    multi = [32, 16, 8, 6, 4, 2, 1]
    
    x, r = block(x, style_input[i], noise_input, nfilters * multi[0], up = False)
    rgbs.append(r)
    
    for i in range(nlayers):
        x, r = block(x, style_input[i], noise_input, nfilters * multi[1+i])
        rgbs.append(r)
        
    x = tf.reduce_sum(rgbs, axis = 0)
    
    return K.Model(inputs = style_input + [noise_input], outputs = x)
    
class StyleGAN2():
    def __init__(self, image_size = 256, batch_size = 8, mapping_layers = 7, mapping_dim = 512, lr = 1e-4):
        self.g_layers = int(np.log2(image_size) - np.log2(4))
        
        self.M = G_map(ndims = mapping_dim, nlayers = mapping_layers)
        self.G = G_synthesis(image_size = image_size, nfilters = 24, nlayers = self.g_layers)
        self.D = Discriminator(filters = 24, image_size = image_size)
        
        self.mapping_dim = mapping_dim
        self.batch_size = batch_size
        self.image_size = image_size
        self.lr = lr
        
        self.pl_mean = 0.50
        self.pl_ema = 0.99
        self.step = 0
        self.history = {'R1': [], 'pl': [], 'G': [], 'D': [], 'step': 0}
        
        self.gen_opt = K.optimizers.Adam(lr = self.lr)
        self.dis_opt = K.optimizers.Adam(lr = self.lr)
        
        self.create_generator()
        
    def create_generator(self):
        noise = L.Input([self.image_size, self.image_size, 1])
        
        inputs, styles = [], []
        
        for i in range(self.g_layers):
            inputs.append(L.Input(shape = (self.mapping_dim,), dtype = tf.float32))
            styles.append(self.M(inputs[-1]))

        self.GM = K.Model(inputs = [inputs, noise], outputs = self.G([styles, noise]))
        
    def generate(self, z = None, noise = None):
        w = []
        if not z: z = np.random.normal(size = (self.batch_size, self.mapping_dim))
        
        for i in range(self.g_layers):
            w.append(self.M(z))
        
        if not noise:
            noise = np.random.uniform(0.0, 1.0, size = [self.batch_size, self.image_size, self.image_size, 1])
        noise = noise.astype('float32')
        
        return self.G([w, noise])
    
    def train(self, steps = 1):
        for t in range(steps):
            if np.random.rand() < 0.5:
                r = np.random.normal(size = (self.batch_size, self.mapping_dim))
                z = [r] * self.g_layers
            else:
                r1 = np.random.normal(size = (self.batch_size, self.mapping_dim))
                r2 = np.random.normal(size = (self.batch_size, self.mapping_dim))
                s = np.random.randint(1, self.g_layers - 1)
                z = [r1] * s + [r2] * (self.g_layers - s)
            
            noise = np.random.uniform(0.0, 1.0, size = [self.batch_size, self.image_size, self.image_size, 1])
            noise = noise.astype('float32')
            
            images = self.data_loader.get_batch()
            
            g, d, pl, r1 = self.train_step(images, z, noise, self.step % 20 == 0)
            g, d, pl, r1 = np.mean(g), np.mean(d), np.mean(pl), np.mean(r1)
            
            if self.step % 20 == 0:
                self.pl_mean = self.pl_ema * self.pl_mean + (1 - self.pl_ema) * pl
            
            print("i:", t, "D: %.3f" % d, "G: %.3f" % g, "PL: %.5f" % self.pl_mean, "      ", end = '\r')
            
            self.step += 1
            
            self.history['R1'].append(r1)
            self.history['pl'].append(self.pl_mean)
            self.history['G'].append(g)
            self.history['D'].append(d)
            self.history['step'] = self.step
            
            if self.step % 10000 == 0:
                self.saveWeights(suffix = "_backup_" + str(self.step))
            
    @tf.function
    def train_step(self, images, z, noise, pl_reg):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            w = []
            for i in range(self.g_layers):
                w.append(self.M(z[i]))
            
            gen_output = self.G([w, noise])
            
            dis_real = self.D(images)
            dis_fake = self.D(gen_output)
            
            gen_loss = tf.reduce_mean(dis_fake)
            dis_loss = tf.reduce_mean(tf.nn.relu(1 + dis_real) + tf.nn.relu(1 - dis_fake))
            
            grad = K.backend.square(K.backend.gradients(dis_real, images)[0])
            r1_loss = tf.reduce_mean(K.backend.sum(grad, axis=np.arange(1, len(grad.shape)))) * 10
            dis_loss += r1_loss
            
            pl_len = 0
            if pl_reg:
                w2 = []
                for i in range(self.g_layers):
                    std = 0.1 / (K.backend.std(w[i], axis = 0, keepdims = True) + 1e-8)
                    w2.append(w[i] + K.backend.random_normal(tf.shape(w[i])) / (std + 1e-8))

                pl_output = self.G([w2, noise])
                pl_len = tf.reduce_mean(K.backend.square(pl_output - gen_output), axis = [1, 2, 3])
                if self.pl_mean > 0:
                    pl_loss = tf.reduce_mean(K.backend.square(pl_len - self.pl_mean))
                    gen_loss += pl_loss
            
            gradients_of_generator = gen_tape.gradient(gen_loss, self.GM.trainable_variables)
            gradients_of_discriminator = dis_tape.gradient(dis_loss, self.D.trainable_variables)
            
            self.gen_opt.apply_gradients(zip(gradients_of_generator, self.GM.trainable_variables))
            self.dis_opt.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))
        
        return gen_loss, dis_loss, pl_len, r1_loss
    
    def saveWeights(self, suffix = ""):        
        self.D.save_weights("D" + str(suffix) + ".h5")
        self.G.save_weights("G" + str(suffix) + ".h5")
        self.M.save_weights("M" + str(suffix) + ".h5")
        np.save("history" + str(suffix) + ".npy", self.history)

    def loadWeights(self, suffix = ""):
        self.D.load_weights("D" + str(suffix) + ".h5")
        self.G.load_weights("G" + str(suffix) + ".h5")
        self.M.load_weights("M" + str(suffix) + ".h5")
        history = np.load("history" + str(suffix) + ".npy")
        self.history['R1'] = history.item().get('R1')
        self.history['pl'] = history.item().get('pl')
        self.history['G'] = history.item().get('G')
        self.history['D'] = history.item().get('D')
        self.history['step'] = history.item().get('step')
        self.step = self.history['step']
        if self.history['pl']:
            self.pl_mean = self.history['pl'][-1]

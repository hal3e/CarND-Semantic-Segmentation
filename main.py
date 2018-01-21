# %%
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import sys


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# %%

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    # Create variables
    vgg_input      = graph.get_tensor_by_name('image_input:0')
    vgg_keep_prob  = graph.get_tensor_by_name('keep_prob:0')
    vgg_layer3_out = graph.get_tensor_by_name('layer3_out:0')
    vgg_layer4_out = graph.get_tensor_by_name('layer4_out:0')
    vgg_layer7_out = graph.get_tensor_by_name('layer7_out:0')

    return vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer6_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # 1x1 convolution
    vgg_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding="SAME")

    vgg_decoder_4 = tf.layers.conv2d_transpose(vgg_1x1, num_classes, 4, 2, padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Use 1x1 convolutions to get the same size in order to combine the layers
    vgg_layer4_out_scaled = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding="SAME", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    vgg_decoder_4 = tf.add(vgg_decoder_4, vgg_layer4_out_scaled)

    vgg_decoder_3 = tf.layers.conv2d_transpose(vgg_decoder_4, num_classes, 4, 2, padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Use 1x1 convolutions to get the same size in order to combine the layers
    vgg_layer3_out_scaled = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding="SAME", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    vgg_decoder_3 = tf.add(vgg_decoder_3, vgg_layer3_out_scaled)

    vgg_decoder_out = tf.layers.conv2d_transpose(vgg_decoder_3, num_classes, 16, 8, padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return vgg_decoder_out
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, optimizer, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    print("--- e:0/{} > mean_loss: init".format(epochs))
    for e in range(epochs):
        mean_loss = 0
        counter = 0
        for images, gt_images in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={input_image: images, keep_prob: 0.75, learning_rate: 0.0003, correct_label: gt_images})

            mean_loss += loss
            counter += 1
            print("   e:{}/{} > batch loss: {:5.3f} > {:3}%".format(e + 1, epochs, loss, int((counter * 100) / (578.0 / batch_size))))

        assert (counter != 0), "No training data available!"
        print("---- e:{}/{} > mean_loss: {:.3f}".format(e + 1, epochs, mean_loss / counter))
tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    # tests.test_for_kitti_dataset(data_dir)
    epochs = 10
    batch_size = 2

    # Download pre-trained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        #  To double the training data each image and gt will be flipped during training
        #  The helper function is updated to generate the flipped images

        print("--- Building the model ---")
        # TODO: Build NN using load_vgg, layers, and optimize function
        vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        # Capture the variables that were already trained
        temp_variables = set(tf.global_variables())

        # Add the new layers
        vgg_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        # Creat the optimizer
        correct_label = tf.placeholder(tf.int8, (None, 160, 576, 2))
        learning_rate = tf.placeholder(tf.float32)

        logits, optimizer, cross_entropy_loss = optimize(vgg_output, correct_label, learning_rate, num_classes)

        # Initialize the variables
        sess.run(tf.variables_initializer(set(tf.global_variables()) - temp_variables))

        saver = tf.train.Saver()

        train_new = True
        # train_new = False

        # fine_tune = True
        fine_tune = False # Load the model and create the predictions

        if train_new:
            # Train the network
            print("\n\n --- Training new network --- \n")
            train_nn(sess, epochs, batch_size, get_batches_fn, optimizer, cross_entropy_loss, vgg_input, correct_label, vgg_keep_prob, learning_rate)

            # Save Model
            save_path = saver.save(sess, "./data/model/model.ckpt")
            print("   Model saved in file: %s" % save_path)
        else:
            # Restore model
            saver.restore(sess, "./data/model/model.ckpt")
            print("   Model restored")

            if fine_tune:
                # Train the network
                print("\n\n --- Fine tunning the network --- \n")
                train_nn(sess, epochs, batch_size, get_batches_fn, optimizer, cross_entropy_loss, vgg_input, correct_label, vgg_keep_prob, learning_rate)

                # Save Model
                save_path = saver.save(sess, "./data/model/model.ckpt")
                print("Model saved in file: %s" % save_path)


        # Save inference data
        helper.save_inference_samples('run', 'data', sess, image_shape, logits, vgg_keep_prob, vgg_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

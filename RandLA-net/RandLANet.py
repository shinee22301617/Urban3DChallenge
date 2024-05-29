from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import tf_util
import time
from tool import Plot

import numpy as np
import os
import sys
import tensorflow as tf
print(tf.VERSION)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/3d_interpolation'))
#from tf_interpolate import three_nn, three_interpolate
#import tf_grouping
#import tf_sampling

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d', time.gmtime())
                self.saving_path = self.saving_path + '_' + dataset.name
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            #self.inputs['sub_features'] = flat_inputs[4 * num_layers:5 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.loss_type = 'sqrt'  # wce, lovas
            self.class_weights = DP.get_class_weights(dataset.num_per_class, self.loss_type)
            self.Log_file = open('log_train_' + dataset.name + '.txt', 'a')

        with tf.variable_scope('layers'):
            self.logits = self.inference(self.inputs, self.is_training)

        with tf.variable_scope('loss'):
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=1)
        c_proto = tf.compat.v1.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):

        d_out = self.config.d_out

        feature = inputs['features']
        #len(inputs['neigh_idx'])) = 5
        # feature.shape = (?, ?, 6)
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)
        # feature.shape = (?, ?, 1, 8)

        # inputs['xyz'][0].shape = (? , ? , 3)
        as_neighbor = 16
        bn_decay = 1.0
        weight_decay = 0
        bn = True
        #grouped_xyz, new_point, idx = self.grouping(feature, 16, inputs['xyz'][0], inputs['xyz'][0], use_knn=True, radius=None) #return neighbors' xyz and features
        #new_xyz, new_feature = self.AdaptiveSampling(grouped_xyz, new_point, as_neighbor, is_training, bn_decay, weight_decay, 'layer'+ str(0), bn=True)
                                   #AdaptiveSampling(grouped_xyz, new_point, as_neighbor, is_training, bn_decay, weight_decay, scope, bn)
        
        #grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, 16, 1])  # translation normalization
        #feature = tf.concat([grouped_xyz, new_point], axis=-1)
        #grouping(feature, nsample, xyz, new_xyz,use_knn=use_knn,radius=radius)
        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self. dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i], 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            feature = f_sampled_i
            #feature1.shape = (?,?,1,32)
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        feature = tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                 'decoder_0', [1, 1], 'VALID', True, is_training)

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1])
            f_decoder_i = tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                   f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                   'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                   is_training=is_training)
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        f_layer_fc1 = tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                     is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])#(?, ?, 13)
        
        return f_out

    def train(self, dataset):
        
        #log_out('***xyz{}***'.format(self.inputs['xyz'][0,:,:]), self.Log_file)
        
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        self.sess.run(dataset.train_init_op)
        
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op,
                       self.extra_update_ops,
                       self.merged,
                       self.loss,
                       self.logits,
                       self.labels,
                       self.accuracy]
                _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
                
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:

                if dataset.use_val and self.training_epoch % 2 == 0:
                    m_iou = self.evaluate(dataset)
                    if m_iou > np.max(self.mIou_list):
                        # Save the best model
                        snapshot_directory = join(self.saving_path, 'snapshots')
                        makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                        self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                    self.mIou_list.append(m_iou)
                    log_out('Best m_IoU of {} is: {:5.3f}'.format(dataset.name, max(self.mIou_list)), self.Log_file)
                else:
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', self.training_step)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])


        print('finished')
        self.sess.close()

    def evaluate(self, dataset):

        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        data_list = self.sess.run(dataset.flat_inputs)
        xyz = data_list[0]
        label = data_list[21]
        a_xyz = np.asarray(xyz)
        a_label = np.asarray(label)
        print(a_xyz.shape) #(10, 30000, 3)
        print(a_label.shape) #(10, 30000)
        
        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0
        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.labels, self.accuracy)
                stacked_prob, labels, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                #if(self.training_epoch >= 75):
                    #p = np.asarray(pred)
                    #print(p.shape)
                    #Plot.draw_pc_sem_ins(a_xyz[0,:,:], p[:,])
                if not self.config.ignored_label_inds:
                    pred_valid = pred
                    labels_valid = labels
                else:
                    invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                    labels_valid = np.delete(labels, invalid_idx)
                    labels_valid = labels_valid - 1
                    pred_valid = np.delete(pred, invalid_idx)

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n] + 0.1)
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training) # LosSE + Attentive Pooling
        f_pc = tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                              activation_fn=None)
        shortcut = tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                  activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training): # LosSE + Attentive Pooling
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        
        '''
        #mod
        f_xyz = tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp3', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_3', is_training)
        '''
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    def PointASNLSetAbstraction(self, xyz, feature, npoint, nsample, mlp, is_training, bn_decay, weight_decay, scope, bn=True, use_knn=True, radius=None, as_neighbor=8, NL=True):
        ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            point: int32 -- #points sampled in Euclidean space by farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
        '''
        with tf.variable_scope(scope) as sc:

            batch_size, num_points, num_channel = feature.get_shape()
            '''Farthest Point Sampling'''
            if num_points == npoint:
                new_xyz = xyz
                new_feature = feature
            else:
                new_xyz, new_feature = sampling(npoint, xyz, feature) #FPS

            grouped_xyz, new_point, idx = self.grouping(feature, nsample, xyz, new_xyz,use_knn=use_knn,radius=radius)
            nl_channel = mlp[-1]

            '''Adaptive Sampling'''
            if num_points != npoint:
                new_xyz, new_feature = self.AdaptiveSampling(grouped_xyz, new_point, as_neighbor, is_training, bn_decay, weight_decay, scope, bn)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
            new_point = tf.concat([grouped_xyz, new_point], axis=-1)

            '''Point NonLocal Cell'''
            if NL:
                new_nonlocal_point = self.PointNonLocalCell(feature, tf.expand_dims(new_feature, axis=1),
                                                    [max(32, num_channel//2), nl_channel],
                                                    is_training, bn_decay, weight_decay, scope, bn)

            '''Skip Connection'''
            skip_spatial = tf.reduce_max(new_point, axis=[2])
            skip_spatial = tf_util.conv1d(skip_spatial, mlp[-1], 1,padding='VALID', stride=1,
                                        bn=bn, is_training=is_training, scope='skip',
                                        bn_decay=bn_decay, weight_decay=weight_decay)

            '''Point Local Cell'''
            for i, num_out_channel in enumerate(mlp):
                if i != len(mlp) - 1:
                    new_point = tf_util.conv2d(new_point, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1],
                                                bn=bn, is_training=is_training,
                                                scope='conv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)


            weight = self.weight_net_hidden(grouped_xyz, [32], scope = 'weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)
            new_point = tf.transpose(new_point, [0, 1, 3, 2]) ##行跟列交換
            new_point = tf.matmul(new_point, weight)
            new_point = tf_util.conv2d(new_point, mlp[-1], [1,new_point.get_shape()[2].value],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training, 
                                            scope='after_conv', bn_decay=bn_decay, weight_decay = weight_decay)

            new_point = tf.squeeze(new_point, [2])  # (batch_size, npoints, mlp2[-1])

            new_point = tf.add(new_point,skip_spatial)

            if NL:
                new_point = tf.add(new_point, new_nonlocal_point)

            '''Feature Fushion'''
            new_point = tf_util.conv1d(new_point, mlp[-1], 1,
                                    padding='VALID', stride=1, bn=bn, is_training=is_training, 
                                    scope='aggregation', bn_decay=bn_decay, weight_decay=weight_decay)

            return new_xyz, new_point

    
    def grouping(self, feature, K, src_xyz, q_xyz, use_xyz=True, use_knn=True, radius=0.2, i=0):
        '''
        K: neighbor size
        src_xyz: original point xyz (batch_size, ndataset, 3)
        q_xyz: query point xyz (batch_size, npoint, 3)
        '''
        batch_size = src_xyz.get_shape()[0]
        #batch_size = 6
        npoint = q_xyz.get_shape()[1]
        #npoint = 30000
        if use_knn:
            point_indices = self.inputs['neigh_idx'][i] #sum 5
            batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, npoint, K, 1))
            idx = tf.concat([batch_indices, tf.expand_dims(point_indices, axis = 3)], axis=3)
            idx.set_shape([batch_size, npoint, K, 2])
            grouped_xyz = tf.gather_nd(src_xyz, point_indices)
        else:
            point_indices, _ = tf_grouping.query_ball_point(radius, K, src_xyz, q_xyz)
            grouped_xyz = tf_grouping.group_point(src_xyz, point_indices)

        #print("---------------------------------")
        #print(feature.shape)  #(?, ?, 3)
        #print(idx.shape)      #(6, 30000, 16, 2)
        #print(tf.gather_nd(feature, idx).shape) #(6, 30000, 16, 3)
        #print("---------------------------------")
        grouped_feature = tf.gather_nd(feature, idx)
        if use_xyz:
            grouped_feature = tf.concat([grouped_xyz, grouped_feature], axis = -1)
        
        return grouped_xyz, grouped_feature, idx
        
    def AdaptiveSampling(self, group_xyz, group_feature, num_neighbor, is_training, bn_decay, weight_decay, scope, bn):
        with tf.variable_scope(scope) as sc:
            #num_channel = group_feature.get_shape()[-1]
            num_channel = 3
            if num_neighbor == 0: #Finish all neighbors
                new_xyz = group_xyz[:, :, 0, :]
                new_feature = group_feature[:, :, 0, :]
                return new_xyz, new_feature
            shift_group_xyz = group_xyz[:, :, :num_neighbor, :]
            shift_group_points = group_feature[:, :, :num_neighbor, :]
            sample_weight = self.SampleWeights(shift_group_points, shift_group_xyz, [32, 1 + num_channel], is_training, bn_decay, weight_decay, scope, bn)
            new_weight_xyz = tf.tile(tf.expand_dims(sample_weight[:,:,:, 0],axis=-1), [1, 1, 1, 3])
            new_weight_feture = sample_weight[:,:,:, 1:]
            new_xyz = tf.reduce_sum(tf.multiply(shift_group_xyz, new_weight_xyz), axis=[2])
            new_feature = tf.reduce_sum(tf.multiply(shift_group_points, new_weight_feture), axis=[2])

            return new_xyz, new_feature

    def SampleWeights(self,new_point, grouped_xyz, mlps, is_training, bn_decay, weight_decay, scope, bn=True, scaled=True):
        """Input
            grouped_feature: (batch_size, npoint, nsample, channel) TF tensor
            grouped_xyz: (batch_size, npoint, nsample, 3)
            new_point: (batch_size, npoint, nsample, channel)
            Output
            (batch_size, npoint, nsample, 1)
        """
        with tf.variable_scope(scope) as sc:
            batch_size, npoint, nsample, channel = new_point.get_shape()
            bottleneck_channel = max(32,channel//2)
            normalized_xyz = grouped_xyz - tf.tile(tf.expand_dims(grouped_xyz[:, :, 0, :], 2), [1, 1, nsample, 1])
            new_point = tf.concat([normalized_xyz, new_point], axis=-1) # (batch_size, npoint, nsample, channel+3)
            transformed_feature = tf_util.conv2d(new_point, bottleneck_channel * 2, [1, 1],
                                                padding='VALID', stride=[1, 1],
                                                bn=bn, is_training=is_training,
                                                scope='conv_kv_ds', bn_decay=bn_decay, weight_decay=weight_decay,
                                                activation_fn=None)
            transformed_new_point = tf_util.conv2d(new_point, bottleneck_channel, [1, 1],
                                                padding='VALID', stride=[1, 1],
                                                bn=bn, is_training=is_training,
                                                scope='conv_query_ds', bn_decay=bn_decay, weight_decay=weight_decay,
                                                activation_fn=None)

            transformed_feature1 = transformed_feature[:, :, :, :bottleneck_channel]
            feature = transformed_feature[:, :, :, bottleneck_channel:]

            weights = tf.matmul(transformed_new_point, transformed_feature1, transpose_b=True)  # (batch_size, npoint, nsample, nsample)
            if scaled:
                weights = weights / tf.sqrt(tf.cast(bottleneck_channel, tf.float32))
            weights = tf.nn.softmax(weights, axis=-1)
            channel = bottleneck_channel

            new_group_features = tf.matmul(weights, feature)
            new_group_features = tf.reshape(new_group_features, (batch_size, npoint, nsample, channel))
            for i, c in enumerate(mlps):
                activation = tf.nn.relu if i < len(mlps) - 1 else None
                new_group_features = tf_util.conv2d(new_group_features, c, [1, 1],
                                                padding='VALID', stride=[1, 1],
                                                bn=bn, is_training=is_training,
                                                scope='mlp2_%d' % (i), bn_decay=bn_decay, weight_decay=weight_decay,
                                                activation_fn=activation)
            new_group_weights = tf.nn.softmax(new_group_features, axis=2)  # (batch_size, npoint,nsample, mlp[-1)
            return new_group_weights

    def weight_net_hidden(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay = None, activation_fn=tf.nn.relu):

        with tf.variable_scope(scope) as sc:
            net = xyz
            for i, num_hidden_units in enumerate(hidden_units):
                net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                    padding = 'VALID', stride=[1, 1],
                                    bn = True, is_training = is_training, activation_fn=activation_fn,
                                    scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

        return net

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg
    

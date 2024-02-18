 article directory 

 + Introduction 

 + Network structure 

 + Feature Learning Network Voxel PartitionGroupingRandom SampllingStacked Voxel Feature EncodingSparse Tensor Representation

Efficient Implementation of Region Proposal Network loss function

 + Training Details 

 + Anchor parameter design positive and negative sample size rule data enhancement

 + experimental results 

 + Evaluation of KITTI validation set Evaluation of KITTI test set

 Thesis: https://arxiv.org/pdf/1711.06396.pdf 

 Code: https://github.com/skyhehe123/VoxelNet-pytorch 

#  brief introduction 

 VoxelNet is an early point cloud detection model proposed by Apple in 2016. It removes the bottleneck of manual feature engineering and proposes a new end-to-end trainability architecture for point cloud-based 3D detection, VoxelNet. VoxelNet can directly operate on sparse 3D points and efficiently capture 3D shape information. Specifically, VoxelNet divides the point cloud into equally spaced 3D voxels and converts a set of points within each voxel into a unified feature representation through a newly introduced voxel feature coding (VFE) layer. In this way, the point cloud is encoded into a descriptive volume representation, which is then connected to an RPN to generate the detection. An efficient implementation of VoxelNet, which benefits from point cloud sparsity and parallel processing on voxel meshes, is also proposed. Our experiments on the KITTI automotive inspection task show that VoxelNet outperforms state-of-the-art 3D detection methods based on LiDAR to a large extent. 

#  network architecture 

 VoxelNet divides the point cloud into equally spaced 3D voxels (Voxels) and transforms a set of points within each voxel into a single feature representation through the newly introduced Voxel Feature Coding (VFE) layer. In this way, the point cloud is encoded into a volume representation with descriptive properties, which is then connected to an RPN to generate detection results. 

 VoxelNet is composed of three functional modules: 

 ![avatar]( 图片.frgpvts27cw.webp) 

 Contribution: 

##  Feature Learning Network 

 The structure of the feature learning network is depicted in the following figure, including Voxel Partition, Grouping, Random Sampling, Stacked Voxel Feature Encoding, Sparse Tensor Representation, and other steps, specifically: 

###  Voxel Partition 

 Given a point cloud, we subdivide 3D space into equally spaced voxels, as shown in Figure 2. Suppose the point cloud contains 3D space with ranges of D, H, W in the Z, Y, X axes, respectively. Define each voxel size as, and. The size of the resulting 3D voxel mesh is,,. Here assume that D, H, W are multiples of,. 

>  Select among papers  

           v 

           D 

          = 

          0.4 

          ， 

           v 

           H 

          = 

          0.2 

          ， 

           v 

           W 

          = 

          0.2 

         v_D = 0.4，v_H = 0.2，v_W = 0.2 

     vD = 0.4, vH = 0.2, vW = 0.2,  

           D 

            ′ 

         D^{'} 

     D′= 10， 

           H 

            ′ 

         H^{'} 

     H′ = 400， 

           W 

            ′ 

         W^{'} 

     W′ = 352。 

###  Grouping 

 The points are grouped according to the voxel where they are located. Due to factors such as distance, occlusion, relative pose of objects, and non-uniform sampling, the LiDAR point cloud is sparse and non-uniform stepwise throughout space. Therefore, after grouping, the voxels contain a variable number of points. As shown in Figure 2, where Voxel-1 has significantly more points than Voxel-2 and Voxel-4, while Voxel-3 has no points. 

###  Random Samplling 

 Typically, a high-precision LiDAR point cloud consists of about 100k points. Directly processing all points not only increases the memory/efficiency burden on the computing platform, but also the variable point density throughout the space may bias the detection. To do this, we randomly draw a fixed number of points T from voxels containing more than T. This sampling strategy serves two purposes, (1) to save computation; and (2) to reduce point imbalances between voxels, thereby reducing sampling bias and adding more variation to training. 

 Typically, a high-precision LiDAR point cloud will contain about 100k points, and if all of these points are processed directly, it will result in: 

 Therefore, after the Grouping operation, it is necessary to randomly sample T points in each non-empty voxel. (less than 0, only T points are sampled if the excess is exceeded), thereby reducing the point imbalance between voxels and adding more variation to training. 

>  Thesis setting  

          T 

         T 

     T = 35 as the maximum number of random sampling points in each non-empty voxel 

###  Stacked Voxel Feature Encoding 

 ![avatar]( 525b3d41d0214c8989caffd9f931c3fc.png) 

 Without loss of generality, use VFE Layer-1 to describe the details. The following figure shows the architecture of VFE Layer 1. Each non-empty Voxel is a point set, defined as: 

 Where the i-th point in the point cloud data is represented, including the XYZ coordinate point and the reflectance. 

 First calculate the average value of the points in voxel V. Then we expand each point with the offset of each point from the mean, as follows: 

 Then, using the method proposed by PointNet, each, through the fully connected network FCN (each fully connected layer consists of linear layer, batch normalization (BN) and linear unit (ReLU).) is mapped to a high-dimensional feature space, and the dimension is also changed from (N, 35, 7) to (N, 35, m) 

 After obtaining the point-wise feature representation, then use element-wise MaxPooling on the pair to obtain the local aggregation feature of the voxel, which can be used to encode the surface shape information contained in the voxel 

 Finally, the local aggregation features and point-wise features are spliced together to obtain the output feature set. All non-empty voxels are encoded in the same way and share the parameters in the fully connected FCN, so that each VFE module contains only one (, /2) parameter matrix. 

 Use VFE-i (,) to represent the i-th VFE layer, which transforms the input features of the dimension into the output features of the dimension. The fully connected FCN layer learns a parameter matrix of size × ( /2) 

 The features output by each voxel through VFE contain the high-dimensional features and aggregated local features of each point in the voxel, so only stacking VFE modules can realize the interaction between the information of each point in the voxel and the local aggregation point information, so that the final feature can describe the shape information of the voxel. 

 The next step is to stack this VFE module to get the complete Stacked Voxel Feature Encoding 

>  The parameters of FC in each VFE module are shared. In the implementation of the original paper, a total of two VFEs VFE-1 (7,32) and VFE-2 (32,128) were stacked, and the final FCN mapped the VFE-2 output to  

           R 

           128 

         \mathbb R^{128} 

     R128 

 After Stacked Voxel Feature Encoding, you can get a (N, 35, 128) feature. You need to perform another FC operation on this feature to fuse the previous point feature and the aggregation feature. The input and output of this FC operation remain unchanged. That is, the obtained tensor is still (N, 35, 128), and then Element-wise Maxpool is performed to extract the most specific representative point in each voxel, and use this point to represent the voxel, that is, (N, 35, 128) -- > (N, 1, 128) 

###  Sparse Tensor Representation 

 In the previous Stacked Voxel Feature Encoding process, non-empty voxels are processed, and these voxels only correspond to a small part of the 3D space. The N non-empty voxel features need to be remapped back into the 3D space and represented as a sparse 4Dtensor, (C, Z ', Y', X ') - > (128, 10, 400, 352). This sparse representation method greatly reduces the memory consumption and the computational consumption in backpropagation. It is also an important step in the implementation of VoxelNet for efficiency. 

 By processing the non-empty voxel lattice through the above process, a series of voxel features can be obtained. This series of voxel features can be represented by a 4-dimensional sparse tensor, the size is, although the point cloud contains about 100k points, more than 90% of the voxels are usually empty. Converting the non-empty voxel features to sparse tensor representation can greatly reduce the memory usage and computational consumption during backpropagation 

##  intermediate convolutional layer 

 After the feature extraction of the Stacked Voxel Feature Encoding layer and the representation of the sparse tensor, 3D convolution can be used for feature extraction between the whole, because the information of each voxel is extracted in the previous VFE. Here, 3D convolution is used to aggregate the local relationship between the voxels, expand the receptive field to obtain richer shape information, and give the subsequent RPN layer to predict the result. 

 Use Conv to represent an M-dimensional convolution operator, where, and is the number of input and output channels, and k, s, and p are the corresponding convolution kernel size, step size, and padding parameters, respectively. Each convolution intermediate layer contains a 3D convolution, a BN layer, and a ReLU layer. The convolution intermediate layer aggregates voxel features in the gradually expanding field to obtain richer shape information 

 To aggregate voxel features, three convolutional intermediate layers are used in turn as Conv3D (128, 64, 3, (2, 1, 1), (1, 1, 1)), Conv3D (64, 64, 3, (1, 1, 1), (0, 1, 1)) and Conv3D (64, 64, 3, (2, 1, 1), (1, 1, 1)) to produce a 4D tensor of size 64 × 2 × 400 × 352 

 After going through the Convolutional middle layers, the data needs to be organized into the specific whole required by the RPN network, and the tensor obtained by the Convolutional middle layers is directly reshaped into (64 * 2,400, 352) in height. Then each dimension becomes C, Y, and X. The reason for this operation is that in the detection tasks of datasets such as KITTI, objects are not stacked in the height direction of 3D space, and there is no such situation as a car being above another car. At the same time, this also greatly reduces the difficulty of designing the RPN layer in the later stage of the network and the number of anchors in the later stage. 

##  Region Proposal Network 

 ![avatar]( 3ab0b144fbcb4924b9c024f2b24e867d.png) 

 Recently, region proposal networks have become an important part of the best performing object detection framework. In this work, several key modifications were made to the RPN architecture proposed in Faster r-cnn and combined with the feature learning network and the convolutional middle layer to form an end-to-end training. The input to the RPN is a feature map of size 128 × 400 × 352, with inches corresponding to the width of the channel, height, and 3D tensor. The network has three fully convolutional layer blocks. The first layer of each block, a convolution with step 2, will undersample the feature map, followed by a convolution with step 1. Each convolution is followed by a BN and RELU layer. Then the features sampled at different scales are deconvolution operations to turn them into feature maps of the same size. Then splice these feature maps from different scales to build high-resolution feature maps for final detection. The final output is a classification prediction result and an anchor regression prediction result. 

 The anchor matching criteria are as follows: If the anchor has the highest intersection (IoU) with ground truth, or if its IoU with ground truth is higher than 0.6 (bird's eye view), the anchor is considered positive. If the IoU between the anchor and all truth boxes is less than 0.45, the anchor is considered negative. 

##  Loss function 

 Let, is the set of positive anchors, is the set of negative anchors. We parameterize a 3D ground truth box to be, where, representing the center position, is the length, width, height box, and is the yaw rotation about the Z axis. To retrieve the ground truth box from the matching positive anchors parameterized to, we define the residual vector containing 7 regression targets corresponding to the center position, < unk > x, < unk > y, < unk > z, three dimensions < unk > l, < unk > w, < unk > h, and rotation < unk >, calculated as follows: 

 Defined as a set of positive samples and a set of negative samples, using a 3D ground truth box representing the truth, which represents the center coordinate of the ground truth, represents the length, width and height of the labeled box, and represents the yaw angle, which represents the predicted positive sample box 

 ![avatar]( 2aaa774327d04257a336b37a6b911b49.png) 

 Encode the loss function for each anchor and GT, containing the seven regression targets, including the center position, for the three dimensions, such as: < unk > x, < unk > y, < unk > z, < unk > l, < unk > w, < unk > h, and rotation < unk >, with the following formula: where the diagonal of the positive sample 

          d 

          a 

        d^a 

    Da is:  

           d 

           a 

          = 

            ( 

             l 

             a 

             ) 

             2 

            + 

            ( 

             w 

             a 

             ) 

             2 

         d^a = \sqrt{ (l^a)^2 + (w^a)^2} 

     da=(la)2+(wa)2

​ 

 ![avatar]( ec5bb39c9050449094fcc2260add9c97.png) 

 The loss function is defined as:  

 The first two terms of the loss function represent the classification loss (which has been normalized) for the positive and negative sample outputs, and the ground truth boxes of neural networks, respectively. The first two terms of the loss function represent the classification loss (which has been normalized) for the positive and negative sample outputs, where they represent cross entropy, and are two constants, which are used as weights to balance the effect of positive and negative sample loss on the final loss function. Represents the regression loss, which uses the Smooth L1 function here. 

##  Efficient implementation 

 The GPU is optimized for dense tensor structures. The point cloud problem is used directly: the points are sparsely distributed in space, and each voxel has a variable number of points. The author has devised a method: a method of transforming a point cloud into a dense tensor structure, where the Stacked VFE operation can be processed in parallel across points and voxels. 

 ![avatar]( 图片.4lid9rl29ibk.webp) 

 Initialize a K × T × 7-dimensional tensor structure to store the voxel input feature buffer, where K is the maximum number of non-empty voxels, T is the maximum number of points per voxel, and 7 is the input encoding dimension of each point. All points are treated randomly. 

 Traversing the entire point cloud data, checking whether the voxel corresponding to the point already has a voxel input feature buffer, this lookup operation is done using a hash table, where the voxel coordinates represent the hash key. 

 The voxel input feature and coordinate buffers can be built by traversing the list of points in a single pass with O (n) time complexity. To further improve memory/computational efficiency, it is possible to store only a limited number (K) of voxels and ignore points from voxels with few points. 

 After building the voxel input feature buffer, Stacked Voxel Feature Encoding only involves point-level and voxel-level dense operations, which can be parallel computing on the GPU. Note that after the stitching operation in VFE, zeroing the features corresponding to empty points guarantees the consistency of the features of the voxel and the features of the points. Finally, using the stored coordinate buffer, the computed sparse voxel structure is reorganized into a dense voxel grid. Through the GPU, the subsequent intermediate feature extraction and RPN layer are completed on the dense voxel grid, 

#  Training Details 

##  Parameter design of anchor 

 Only one anchor scale is used for each category, namely car [3.9, 1.6, 1.56], center at -1 meters, person [0.8, 0.6, 1.73], center at -0.6 meters, bicycle [1.76, 0.6, 1.73], center at -0.6 meters, and two orientation information is added for each anchor to rotate 0 degrees and 90 degrees around the center. 

##  Positive and negative sample size rule 

 In the process of anchor matching GT, the 2D IOU match type is used, which is matched directly from the generated feature map, that is, the BEV perspective; there is no need to consider the height information. There are two reasons: 1. Because all the objects in the kitti dataset are in the same plane in 3D space, there is no case where the car is on the car. 2. The height difference between all categories of objects is not very large, and a good result can be obtained directly by using SmoothL1 regression. Secondly, the iou threshold for each anchor is set as a positive and negative sample is: 

 The vehicle matching Iou threshold is greater than or equal to 0.6 for positive samples, less than 0.45 for negative samples, and no loss is calculated in the middle. 

 The human matching Iou threshold is greater than or equal to 0.5 for positive samples, less than 0.35 for negative samples, and the intermediate loss is not calculated. 

 Bicycle matching Iou thresholds greater than or equal to 0.5 are positive samples, less than 0.35 are negative samples, and the intermediate loss is not calculated. 

##  data augmentation 

 With fewer than 4000 training point clouds, training our network from scratch will inevitably suffer from overfitting. To reduce this problem, we introduce three different forms of data augmentation. The augmented training data is generated instantaneously and does not need to be stored on disk [20]. 

 We define a set, which is the entire point cloud, consisting of N points. We parameterize the 3D bounding box as, where, is the center position, is the length, width, height, and is the yaw rotation around the Z axis. We define, as a set containing all LiDAR points in b i, where, represents a specific LiDAR point in the entire set M. 

#  Experimental results 

 Evaluate VoxelNet on the KITTI 3D Object Detection Benchmark, which contains 7,481 training images/point clouds and 7,518 test images/point clouds covering three categories: cars, pedestrians, and bicycles. For each category, the detection results are evaluated based on three difficulty levels: Easy, Medium, and Hard, which are determined based on object size, occlusion state, and truncation level. Since the ground truth of the test set is unavailable and access to the test server is limited, the training data is subdivided into a training dataset and a validation set, with 3712 data samples for training and 3769 data samples for validation. Splitting avoids samples from the same sequence being included in both the training and validation sets. 

 For the automotive category, the proposed method is compared with several of the best performing algorithms, including image-based methods: Mono3D and 3DOP; LiDAR-based methods: VeloFCN and 3D-FCN; and multimodal methods MV. 

 To analyze the importance of end-to-end learning, the author implements a hand-crafted baseline (HC-baseline), derived from the VoxelNet architecture, using hand-crafted features 

 ##  

##  Evaluation of the KITTI validation set 

 ![avatar]( 78fa23770de741e099ad968e655a9783.png) 

 Metrics IoU thresholds are 0.7 for the Car class and 0.5 for the Pedestrian and Cyclist classes. IoU thresholds are the same for bird's-eye views and full 3D assessments. Use the Average Precision (AP) metric to compare these methods. 

 Evaluation in Bird's Eye View results are shown in Table 1. VoxelNet consistently outperformed all competing methods on all three difficulty levels. 

 Evaluation in 3D Table 2 summarizes the comparisons. For the Car class, VoxelNet AP significantly outperformed other methods at all difficulty levels. Specifically, using LiDAR alone, VoxelNet significantly outperformed the state-of-the-art method MV (BV + FV + RGB) based on LiDAR + RGB at the simple, medium, and difficulty levels, by 10.68%, 2.78%, and 6.29%, respectively. 

 As with the aerial view evaluation, VoxelNet was also compared to HC-baseline for 3D pedestrian and bicycle detection. Because of the height variation in 3D pose and shape, successful detection of both categories requires better 3D shape representation. Table 2 shows that VoxelNet can be better used for the challenging 3D detection task VoxelNet and is more effective than hand-crafted features in capturing 3D shape information. 

##  Evaluation of the KITTI test set 

 ![avatar]( 图片.69l4p4kbj4lc.webp) 

 ![avatar]( 77d79ac8bfe04dd695f3a4a9f57fb228.png) 

 VoxelNet was evaluated on the KITTI test set by submitting the test results to the official server, and the results are shown in Table 3. VoxelNet outperformed the previous state-of-the-art MV significantly in all tasks (aerial view and 3D detection), and VoxelNet only used LiDAR. 

 Several examples of 3D detection are shown above in Figure 6. For better visualization, 3D boxes detected using LiDAR are projected onto an RGB image. As shown, VoxelNet provides highly accurate 3D bounding boxes in all categories. 

 On TitanX GPUs and 1.7Ghz CPUs, the inference time of VoxelNet is 225ms, of which the voxel input feature computation takes 5ms, the feature learning network takes 20ms, the convolutional middle layer takes 170ms, and the region proposal network takes 30ms. 


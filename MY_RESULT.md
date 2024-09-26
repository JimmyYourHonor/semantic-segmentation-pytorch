# Positional embedding project

This project builds upon the PyTorch implementation of semantic segmentation models on MIT ADE20K scene parsing dataset (http://sceneparsing.csail.mit.edu/). And the semantic segmentation architechture Segformer (https://arxiv.org/pdf/2105.15203).

First, I have used simple pytorch code to reimplement segformer instead of using the mmcv library. So, I can run experiments easily with Google Colab. To get comparable or even better accuracy as the original paper, I not only implemented the training pipeline but also added deminishing momentum to improve accuracy.

Seoncd, the Segformer model uses convolution in feed forward layer and doesn't use any positional encoding in the network. This gives the network shift invariant property. For my experiment, I added implementation of rotary positional embedding as a variant of segformer. After finetining the pretrained model mit_b0, I got better accuracy than the original model on single scale evaluation but not on multi-scale evalution. Adding postional embedding, the network shows some potential. However, when we run inference on different input size as training and interpolate the positional encoding, it's not as adaptive as the original method.

Third, I tried to use another method. I implemented a sliding window attention module so that the positional encoding is the same in traing time and test time. This variant shows better accuracy in both single scale and multiscale. However, the inference time for large sliding window size and small stride is long.

<table><tbody>
    <th valign="bottom">Architecture</th>
    <th valign="bottom">MultiScale Testing</th>
    <th valign="bottom">Mean IoU</th>
    <th valign="bottom">Pixel Accuracy(%)</th>
    <tr>
        <td rowspan="2">Segformer B0</td>
        <td>No</td><td>37.38</td><td>77.76</td>
    </tr>
    <tr>
        <td>Yes</td><td>38.44</td><td>78.74</td>
    </tr>
    <tr>
        <td rowspan="2">Segformer B0 w/ pos</td>
        <td>No</td><td>37.68</td><td>77.87</td>
    </tr>
    <tr>
        <td>Yes</td><td>38.30</td><td>78.65</td>
    </tr>
    <tr>
        <td rowspan="2">Segformer B0 w/ pos and slide</td>
        <td>No</td><td>37.6</td><td>77.5</td>
    </tr>
    <tr>
        <td>Yes</td><td>38.96</td><td>79.06</td>
    </tr>
</tbody></table>
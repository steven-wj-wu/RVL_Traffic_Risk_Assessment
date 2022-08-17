Here are the codes of train and test the Driver Attention Model.
The code is based on CDNN:traffic driving saliency & eye tracking dataset:
https://github.com/taodeng/CDNN-traffic-saliency
The data load is changed to load visual ground truth images from BDDA dataset(https://bdd-data.berkeley.edu/) and optical flow information.
the attention gate is add to the origin code. We trained the CDNN weights and attention gate weights  separately. 

# Lane Detection - CIS 581 Final Project
Lejun Jiang, Han Shao, Ying Yang

## Dataset
For this project, we use TuSimple Benchmark Dataset. 

Link to the dataset: https://github.com/TuSimple/tusimple-benchmark/issues/3

The test images are from test_set.zip, and the ground truth of those test images is stored in
test_label.json.

## Edge Detection
You can use either your own PC/laptop or Colab.

Before running canny_edge.py, you need to create a **Results** folder 
under laneDetection folder where the result images will be stored.

To run this program on Colab, you should use
```commandline
!git clone https://github.com/hsha0/laneDetection.git
%cd laneDetection/
!mkdir Results
```
```
!python3 canny_edge.py IMAGE_NAME
```
i.e.
```commandline
!python3 canny_edge.py 1.jpg
```

We provide five test images in Test_Images folder, and you can add any other 
test images to it.

To run this program on a local machine, it's simply
```commandline
mkdir Results
python3 canny_edge.py IMAGE_NAME
```

## To Do

- [ ] Tuning low/high of Canny Edge.
- [ ] Write Evaluation function for Edge Detection.
- [ ] Use CuPy to replace numpy to improve efficiency.

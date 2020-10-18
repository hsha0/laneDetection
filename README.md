# Lane Detection - CIS 581 Final Project
Lejun Jiang, Han Shao, Ying Yang

## Dataset
For this project, we use TuSimple Benchmark Dataset. 

Link to the dataset: https://github.com/TuSimple/tusimple-benchmark/issues/3

The test images are from test_set.zip, and the ground truth of those test images is stored in
test_label.json.

## Edge Detection
You can use Colab (with GPU) to run this program.

Please follow these steps before running the canny_edge.py.
1. install cupy.
2. git clone repository.
3. cd repo folder.
4. make new dictionary **Results**.

Then, you can run canny_edge.py through
```commandline
!pip3 install cupy-cuda101 
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

## To Do

- [ ] Tuning low/high of Canny Edge.
- [ ] Write Evaluation function for Edge Detection.
- [x] Use CuPy to replace numpy to improve efficiency.

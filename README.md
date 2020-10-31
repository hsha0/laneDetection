# Lane Detection - CIS 581 Final Project
Lejun Jiang, Han Shao, Ying Yang

## Dataset
For this project, we use TuSimple Benchmark Dataset. 

Link to the dataset: https://github.com/TuSimple/tusimple-benchmark/issues/3

The test images are from test_set.zip, and the ground truth of those test images is stored in test_label.json.

## Edge Detection
You can use Colab (with GPU) to run canny_edge.ipynb.

Run each cell in order to make sure everything works. Detailed instruction in .ipynb.

## To Do

- [ ] Write Evaluation function for Edge Detection.
- [x] Tuning low/high of Canny Edge.
- [x] Tuning HoughLinesP function.
- [x] Define a region of interest (hardcode? or adaptive?)
- [x] Use CuPy to replace numpy to improve efficiency.

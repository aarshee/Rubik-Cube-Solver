# Rubik's Cube Solver

Solving 3 * 3 Rubik's Cube using Deep Q-Learning

## Install

This project requires **Python 3.5**

### Libraries

This project requires following libraries:

```
Numpy(>=1.14.2)
Tensorflow(>=1.7.0)
Keras(>=2.1.1)
```

### Code

It consists of two python files **rubik.py** and **solver.py**. To solve the Rubik's Cube run 

```
python3 solver.py
```

Rubik's Cube is emulated through

```
python3 rubik.py
```

On running **solver.py** it will pick up the trained model **rubik.final.h5** and report the results. 

*See ***report.pdf*** for complete analysis of the project.
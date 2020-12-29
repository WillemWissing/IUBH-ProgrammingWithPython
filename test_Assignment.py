import unittest
import pandas as pd
import numpy as np

import Assignment


class Testclass:
    def __init__(self, x_variable="x"):
        self.x_variable = x_variable
        self.scores = pd.DataFrame()
        self.bestfit = pd.DataFrame()
        self.data = pd.DataFrame()
        self.y_variable = "y"
        self.functions = []


def setup_data():
    global training
    training = Testclass()
    global ideal
    ideal = Testclass()
    global test
    test = Testclass()

    training.data = pd.DataFrame(
        {
            "x": list(range(-5, 6)),
            "y1": list(range(-5, 6)),
            "y2": list(range(-10, 12, 2)),
        }
    )

    ideal.data = pd.DataFrame(
        {
            "x": list(range(-5, 6)),
            "y1": list(range(-4, 7)),
            "y2": list(range(-11, 11, 2)),
            "y3": list(range(0, 11)),
        }
    )

    test.data = pd.DataFrame(
        {
            "x": list(range(-5, 6)),
            "y": list(np.arange(-4.5, 6.5, 1)),
            "Delta Y (test func)": np.nan,
            "Ideal Func #": np.nan,
            "Matched Value": np.nan,
        }
    )


trainingmerged = pd.DataFrame(
    {
        "x": list(range(-5, 6)),
        "y1 (training func)": list(range(-5, 6)),
        "y2 (training func)": list(range(-10, 12, 2)),
        "y1": list(range(-4, 7)),
        "y2": list(range(-11, 11, 2)),
        "y3": list(range(0, 11)),
    }
)

trainingscores = pd.DataFrame(
    {
        "y1 (training func)": [["y1", 11], ["y2", 121], ["y3", 275]],
        "y2 (training func)": [["y1", 121], ["y2", 11], ["y3", 385]],
    }
)

trainingfit = pd.DataFrame(
    {
        "training func": ["y1 (training func)", "y2 (training func)"],
        "ideal func": ["y1", "y2"],
        "score": [11, 11],
    }
)

trainingdelta = pd.DataFrame(
    {
        "training func": ["y1 (training func)", "y2 (training func)"],
        "ideal func": ["y1", "y2"],
        "score": [11, 11],
        "max deviations": [1, 1],
    }
)

testmatch = pd.DataFrame(
    {
        "x": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 1, 2],
        "y": [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 1.5, 2.5],
        "Delta Y (test func)": [
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            -0.5,
        ],
        "Ideal Func #": [
            "y1",
            "y1",
            "y1",
            "y1",
            "y1",
            "y1",
            "y2",
            "y2",
            "y1",
            "y1",
            "y1",
            "y1",
            "y1",
        ],
        "Matched Value": [-4, -3, -2, -1, 0, 1, 1, 3, 4, 5, 6, 2, 3],
        "y1": [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 2, 3],
        "y2": [-11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 1, 3],
    }
)
testmatch["Matched Value"] = testmatch["Matched Value"].astype("float64")


class TestAssignment(unittest.TestCase):
    def test_merge(self):
        setup_data()
        dataset = Assignment.Training.mergedata(training, "x", ideal)
        pd.testing.assert_frame_equal(trainingmerged, dataset)

    def test_score(self):
        setup_data()
        scores = Assignment.Training.leastsquares(
            training, training.data, trainingmerged, training.scores
        )
        pd.testing.assert_frame_equal(trainingscores, scores)

    def test_fit(self):
        setup_data()
        bestfit = Assignment.Training.fit(training, trainingscores, training.bestfit)
        pd.testing.assert_frame_equal(trainingfit, bestfit)

    def test_delta(self):
        setup_data()
        delta = Assignment.Training.delta(training, trainingdelta, trainingmerged)
        pd.testing.assert_frame_equal(trainingdelta, delta)

    def test_find_match(self):
        setup_data()

        data = Assignment.Test.find_match(
            test,
            test.data,
            test.x_variable,
            test.y_variable,
            trainingdelta,
            ideal.data,
            ideal.x_variable,
            test.functions,
        )
        print(data)
        print(testmatch)
        pd.testing.assert_frame_equal(testmatch, data)

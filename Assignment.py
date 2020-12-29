"""
The goal of this program is to select the closest match between various data sets. 

The three data sets, Training (A), Ideal(B) and Test(C), will represent one or several functions, via x-y pair values. 
Dataset A will contain 4 functions, B will contain 50 functions, and C will contain 1 function.

The 4 functions in dataset B that most closely match functions in dataset A will be used to map individual points in dataset C. 

Between dataset A and B, the sum of all y-deviations squared (Least-Square) will be used as a measure for best fit. 
For each datapoint in C, the deviation between C and the function from B must not exceed the maximum deviation between the function in B and its match A by a factor of √2.

Finally, visualizations are generated, and the datasets and the matchings are written to a sql database.


"""

import itertools

import numpy as np
import pandas as pd

from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.palettes import Dark2_5 as palette
from bokeh.plotting import figure, output_file, show

from sqlalchemy import create_engine


class Data:

    """
    A class to load a dataset

    ...

    Attributes
    ----------
    x_variable:
        The variable/column that represents the x value
    location:
        Location on disk where the dataset is to be loaded from
    data:
        The dataset that has been loaded

    """

    def __init__(self, location, x_variable="x"):
        self.x_variable = x_variable
        self.location = location
        self.data = self.load_file(location, x_variable)

        """
        Constructs all the necesssary attributes for the Data object.
        
        Parameters
        ----------
              name: 
                The name of the (set of) function(s)
              x_variable:
                The variable/column that represents the x value
              location:
                 Location on disk where the dataset is to be loaded from
              data:
                The dataset that has been loaded

        Methods
        --------
            load_file:
                Load the data into a dataframe

        """

    def load_file(self, location, x_variable):

        """
        Load the data into a dataframe

        The data is loaded into a pandas dataframe. All columns containing empty values are dropped.
        If the given x value is not found, or if it is the only data present, returns an error.

        """

        try:
            data = pd.read_csv(location)

            for column in data:
                if data[column].dtype != float:
                    data.drop(
                        [column], axis=1, inplace=True
                    )  # drop all columns that contain non-float values

            if data.isnull().any().any():
                print("Data file contains empty cells")

            if x_variable not in data.columns:
                print("X variable not found in data file")

            if len(data.columns) < 2:
                print("No Y values found in data file")

        except Exception as e:
            print(e)
            quit()
        else:
            return data


class Training(Data):

    """
    Class to create a Training data object, inheriting from the Data class

    Attributes
    ----------
        scores:
            The score for a function matched with another function, based on the sum of differences squared
        bestfit:
            A table of each function and its matching ideal function,
            including the scores and the maximum deviation
        dataset:
            A merged table of the training dataset and the ideal dataset

    Methods
    -------
        mergedata:
            Create an inner join with a second dataset
        leastsquares:
            Calculate the sum of differences squared between the datasets
        fit:
            Find the best fit function for each training set
        delta:
            Find the maximum deviation for each fit pairing of data sets
        plot_figures:
            Create overlaying plot of each function pairing
        to_database:
            Create a table in the database with the training dataset
        match_ideal:
            Runs functions in class to match, fit, plot, and export data


    """

    def __init__(self, location, x_variable="x"):

        """
        Constructs all the necesssary attributes for the Training data object.

        Attributes
        ----------
        scores:
            The score for a function matched with another function, based on the sum of differences squared
        bestfit:
            A table of each function and its matching ideal function,
            including the scores and the maximum deviation
        dataset:
            A merged table of the training dataset and the ideal dataset

        """

        super().__init__(location, x_variable)
        self.scores = pd.DataFrame()
        self.bestfit = pd.DataFrame()
        self.dataset = pd.DataFrame()

    def mergedata(self, x_variable, idealfunc):
        """
        Create an inner join with a second dataset on the x functions

        """
        ideal_x_variable = idealfunc.x_variable
        ideal_data = idealfunc.data

        dataset = pd.merge(
            self.data,
            ideal_data,
            left_on=x_variable,
            right_on=ideal_x_variable,
            how="inner",
            suffixes=(" (training func)", None),
        )
        return dataset

    def leastsquares(self, data, dataset, scores):
        """
        Calculate the total sum of the difference between points in two datasets, squared.

        This is calculated for every permutation of a pairing of a training data and an ideal dataset.

        These scores are stored in self.scores

        """

        for i in range(
            1, len(data.columns)
        ):  #  iterate over every column in the training dataset
            training_col = dataset.iloc[:, i]
            squarescores = []
            for j in range(
                len(data.columns), len(dataset.columns)
            ):  #  iterate over every column in the Ideal dataset, which start after the training columns
                ideal_col = dataset.iloc[:, j]
                sumofsquares = (
                    (ideal_col - training_col) ** 2
                ).sum()  #  calculate the difference between the values in each dataset,
                #  square it, and add it to the score calculated so far
                squarescores.append([dataset.columns[j], sumofsquares])
            scores[dataset.columns[i]] = squarescores
        return scores

    def fit(self, scores, bestfit):

        """
        Find the best fitting ideal dataset for each training dataset, using the scores calculated.
        The result of this matching is stored in self.bestfit

        """

        for func in scores:
            for i in range(len(scores[func])):
                if scores[func][i][1] == min(
                    z[1] for z in scores[func]
                ):  #  search the list of scores for the lowest score for each function
                    bestfit = bestfit.append(
                        pd.DataFrame(
                            [[func, scores[func][i][0], scores[func][i][1]]],
                            columns=[
                                "training func",
                                "ideal func",
                                "score",
                            ],  # keep a mapping of the training function and the ideal function
                            # with the lowest score, and the score
                        )
                    )
        bestfit.reset_index(drop=True, inplace=True)
        return bestfit

    def delta(self, bestfit, dataset):
        """
        Calculate the maximum deviation for each matched function to the training function.
        This value is stored in self.bestfit

        """

        bestfit["max deviations"] = list(
            pd.DataFrame(
                dataset[
                    bestfit["ideal func"]
                ].values  # subtract  the training function from the ideal function
                - dataset[
                    bestfit["training func"]
                ].values  # and find the absolute difference
            )
            .abs()
            .max(axis=0)
        )
        return bestfit

    def plot_figures(self, bestfit, dataset, x_variable):

        """
        Create plots of the training function and matched ideal function

        one plot displays both functions as a function of x

        One plot displays the pairings of Y values of both functions, with a reference line of a perfect fit

        """

        for match in bestfit.iterrows():  # iterate over the entries in bestfit
            output_file(
                match[1][0] + ".html"
            )  # Use the function name to create a filename
            x = dataset[x_variable]
            y_train = dataset[match[1][0]]
            y_ideal = dataset[match[1][1]]

            fig1 = figure(
                plot_width=400,
                plot_height=400,
                x_axis_label="X value",
                y_axis_label="Y value",
                title="Training function and Ideal function",
            )
            fig1.scatter(
                x,
                y_train,
                size=2,
                color="blue",
                legend_label=match[1][0],
                marker="circle",
            )
            fig1.scatter(
                x,
                y_ideal,
                size=2,
                color="red",
                legend_label=match[1][1],
                marker="triangle",
            )
            fig1.legend.location = "top_left"

            fig2 = figure(
                plot_width=400,
                plot_height=400,
                x_axis_label="Y value training function",
                y_axis_label="Y value Ideal function/Reference",
                title="Training function vs Ideal function",
            )
            fig2.scatter(
                y_train,
                y_ideal,
                size=2,
                color="purple",
                legend_label=match[1][0] + " vs " + match[1][1],
                marker="triangle",
            )
            fig2.scatter(
                y_train,
                y_train,
                size=2,
                color="black",
                legend_label="Reference line",
                marker="circle",
            )
            fig2.legend.location = "top_left"

            grid = gridplot(children=[[fig1, fig2]])
            show(grid)

    def to_database(self, data, sqlite_connection):
        """
        Create tables in the database with the training data
        """

        sqlite_table = "TrainingDB"
        data.to_sql(sqlite_table, sqlite_connection, if_exists="replace")

    def match_ideal(self, idealfunc, sqlite_connection):

        """
        Runs functions in class to match, fit, plot, and export data

        """
        self.dataset = self.mergedata(self.x_variable, idealfunc)
        self.scores = self.leastsquares(self.data, self.dataset, self.scores)
        self.bestfit = self.fit(self.scores, self.bestfit)
        self.bestfit = self.delta(self.bestfit, self.dataset)
        self.plot_figures(self.bestfit, self.dataset, self.x_variable)
        self.to_database(self.data, sqlite_connection)

        return


class Ideal(Data):
    """
    A class to create an ideal data object, inherited from the data class.

    """

    def __init__(self, location, x_variable="x"):
        super().__init__(location, x_variable)

    def to_database(self, sqlite_connection):

        """
        Create tables in the database with the Ideal data

        """

        sqlite_table = "IdealDB"
        self.data.to_sql(sqlite_table, sqlite_connection, if_exists="replace")


class Test(Data):

    """
    Class to create a Test data object, inheriting from the Data class

    Attributes
    -----------
    y_variable:
        Column containing the Y values
    data["Ideal Func #"]:
        New column in the data attribute containing the # corresponding to the matched Ideal function
    data["Delta Y (test func)"]:
        New column in the data attribute containing the difference between the test datapoint
        and the matched ideal function datapoint
    functions:
        List containing all the matched functions

    Methods:
    --------
    find_match:
        Merge dataset with the functions from the ideal dataset. For each point in original dataset,
        find the closest point in one of the given ideal datasets
    plot_figures:
        Create a plot with the original dataset (black) and all the ideal datasets
    to_database:
        Create a table in the database containing the original dataset,
        the matched function for each point and its deviation
    match_test:
        Runs functions in class to match, fit, plot, and export data



    """

    def __init__(self, location, x_variable="x"):

        """
        Constructs all the necessary attributes for the Test object.

        Parameters
        ----------
        y_variable:
            Column containing the Y values
        data["Ideal Func #"]:
            New column in the data attribute containing the # corresponding to the matched Ideal function
        data["Delta Y (test func)"]:
            New column in the data attribute containing the difference between
            the test datapoint and the matched ideal function datapoint
        functions:
            List containing all the matched functions

        """

        super().__init__(location, x_variable)
        self.y_variable = self.data.columns.drop(x_variable)[0]
        self.data["Delta Y (test func)"] = np.nan
        self.data["Ideal Func #"] = np.nan
        self.data["Matched Value"] = np.nan
        self.functions = []

    def find_match(
        self,
        data,
        x_variable,
        y_variable,
        fit_functions,
        ideal_data,
        ideal_x_variable,
        functions,
    ):

        """
        Iterate through the functions that matched in training.
        Left join the data from these functions to the test dataset (left join to keep all datapoints in the test set).
        Search through the Ideal dataset for matching data (difference no bigger than sqrt(2) * max deviation found in training).
        If a value has multiple matches, the new matches are copied to a new row.
        """

        for (
            item
        ) in (
            fit_functions.iterrows()
        ):  # iterate over all functions found in training.bestfit
            row = item[1]
            ideal_func = row[1]  # select the matched ideal function
            functions.append(
                ideal_func
            )  # add the function name to list of functions, having an attribute with all used functions in a list is usefull for future computations
            data = data.merge(  # merge the function to the data
                ideal_data[[ideal_x_variable, ideal_func]],
                left_on=x_variable,
                right_on=ideal_x_variable,
                how="left",
                suffixes=(" (test func)", None),
            )

            max_deviation = row[3] * np.sqrt(2)  # set the maximum difference allowed
            # to max deviation found in training, multiplied by sqrt(2)

            delta = data[y_variable] - data[ideal_func]  # calculate the difference
            functionfit = (
                delta.abs() <= max_deviation
            )  # check the absolute value of the difference vs the allowed value

            empty_values = data[
                "Delta Y (test func)"
            ].isna()  # check which data points are not matched yet
            duplicate_values = functionfit & np.invert(
                empty_values
            )  # select found matches already have value assigned

            append_data = data[duplicate_values]  # copy the old matches

            data.loc[
                functionfit, ["Delta Y (test func)"]
            ] = delta  # insert value, deviation and matched function, overwriting any previously found matches
            data.loc[functionfit, ["Ideal Func #"]] = ideal_func
            data.loc[functionfit, ["Matched Value"]] = data[ideal_func]
            data = data.append(append_data)  # re-add the older matches
            data.reset_index(inplace=True, drop=True)

        return data

    def plot_figures(self, data, functions, x_variable, y_variable):

        """
        Create various plots with the original dataset and the ideal datasets or the matched datapoints.

        figure 3:
            A plot of the Test dataset and all 4 ideal datasets


        figure 4:
            A plot of the test dataset and the datapoints from the Ideal sets that match.


        figure 5:
            A plot of the matched Ideal data points vs a reference match line ([ytest,ymatched] vs [ytest,ytest])

        """

        output_file("Test_data_matches.html")
        source = ColumnDataSource(
            data=data[[x_variable, *functions]]
        )  # create a data source for our bokeh plot that includes the x column and all function columns
        colors = itertools.cycle(palette)

        fig3 = figure(
            x_axis_label="X value",
            y_axis_label="Y value",
            title="Test function and all Ideal function matches",
        )
        for y_value, color in zip(functions, colors):
            fig3.scatter(
                source=source,
                x=x_variable,
                y=y_value,
                size=4,
                color=color,
                legend_label=y_value,
            )
        fig3.scatter(
            x=data[x_variable],
            y=data[y_variable],
            size=6,
            color="black",
            legend_label="Test Data",
            marker="triangle",
        )
        fig3.legend.location = "top_center"

        fig4 = figure(
            x_axis_label="X value",
            y_axis_label="Y value",
            title="Test function and matched points",
        )
        fig4.scatter(
            x=data[x_variable].loc[
                data["Matched Value"].isna()
            ],  # select only the datapoints from the test function that have no match
            y=data[y_variable].loc[data["Matched Value"].isna()],
            size=6,
            color="black",
            legend_label="Unmatched Test Data",
            marker="triangle",
        )
        fig4.scatter(
            x=data[x_variable],
            y=data["Matched Value"],
            size=8,
            color="red",
            legend_label="Ideal Data",
            marker="diamond",
        )
        fig4.scatter(
            x=data[x_variable].loc[
                data["Matched Value"].isna()
                == False  # select only the datapoints from the test function that have a match
            ],
            y=data[y_variable].loc[data["Matched Value"].isna() == False],
            size=4,
            color="blue",
            legend_label="Matched Test Data",
            marker="circle",
        )

        fig5 = figure(
            x_axis_label="Y value Test Function",
            y_axis_label="Y value Matched Function/Reference",
            title="Test function vs Matched points",
        )
        fig5.scatter(
            x=data[y_variable],
            y=data[y_variable],
            size=6,
            color="black",
            legend_label="Reference Line",
            marker="circle",
        )
        fig5.scatter(
            x=data["Matched Value"],
            y=data[y_variable],
            size=6,
            color="red",
            legend_label="Matched Ideal Data",
            marker="triangle",
        )

        grid = gridplot(children=[[fig3, fig4, fig5]])
        show(grid)

    def to_database(self, data, sqlite_connection):
        """
        Create a table in the database containing the original dataset,
        the matched function for each point and its deviation

        """
        sqlite_table = "TestDB"
        data.iloc[:, :4].to_sql(
            sqlite_table, sqlite_connection, if_exists="replace"
        )  # write only the relevant data to the table

    def match_test(self, training, ideal, sqlite_connection):
        """
        Runs functions in class to match, fit, plot, and export data

        """
        self.data = self.find_match(
            self.data,
            self.x_variable,
            self.y_variable,
            training.bestfit,
            ideal.data,
            ideal.x_variable,
            self.functions,
        )
        self.plot_figures(self.data, self.functions, self.x_variable, self.y_variable)
        self.to_database(self.data, sqlite_connection)


def match_and_map(
    training_location,
    ideal_location,
    test_location,
    training_x="x",
    ideal_x="x",
    test_x="x",
):

    """
    Create a database and use a training, ideal and test dataset to find the best matches,
    create plots and write values into tables in the database

        parameters:
        training_location:
            location on disk of the training data file
        ideal_location:
            name on disk of the ideal data file
        test_location:
            name on disk of the test data file
        training_x:
            name of the x variable in the training dataset, default = 'x'
        ideal_x:
            name of the x variable in the ideal dataset, default = 'x'
        test_x:
            name of the x variable in the test dataset, default = 'x'



    """

    engine = create_engine("sqlite:///function_match_data.db", echo=False)
    sqlite_connection = engine.connect()

    training = Training(training_location, training_x)
    ideal = Ideal(ideal_location, ideal_x)
    test = Test(test_location, test_x)

    training.match_ideal(ideal, sqlite_connection)

    ideal.to_database(sqlite_connection)

    test.match_test(training, ideal, sqlite_connection)


if __name__ == "__main__":  # making sure the program doesn't run while testing
    match_and_map(
        "train.csv",
        "ideal.csv",
        "test.csv",
    )
